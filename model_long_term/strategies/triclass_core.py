"""Core implementation of the ConvGRU-based triclass long-term strategy.

This module centralises model loading, feature scaling and trading logic so that
batch backtests and one-off CLI experiments can reuse the exact same code path.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from model_triclass import ConvGRUAttentionTriclass
from data_loader_triclass import FEATURE_COLS


def normalize_stock_code(code: str) -> str:
    """Normalise stock codes to 6-digit strings for consistent lookups."""
    code_str = str(code).strip()
    if code_str.isdigit() and len(code_str) <= 6:
        return code_str.zfill(6)
    if code_str.endswith((".SH", ".SZ")) and code_str[:-3].isdigit():
        return code_str[:-3].zfill(6)
    if code_str[-6:].isdigit():
        return code_str[-6:]
    return code_str


class TriclassStrategy:
    """Unified long-term trading strategy driven by Conv1d→GRU→Attention model."""

    def __init__(
        self,
        model_path: str = "model_triclass_alpha.pth",
        scaler_path: str = "scaler_triclass.pkl",
        default_config: dict | None = None,
        stock_configs: dict | None = None,
        classification_configs: dict | None = None,
        stock_classification_map: dict | None = None,
        position_config: dict | None = None,
        cost_config: dict | None = None,
    ) -> None:
        base_dir = Path(__file__).resolve().parent.parent

        self.model_path = self._resolve_resource_path(model_path, base_dir)
        self.scaler_path = self._resolve_resource_path(scaler_path, base_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ConvGRUAttentionTriclass(
            in_feats=24,
            conv_channels=64,
            conv_k=5,
            rnn_hidden=128,
            rnn_layers=1,
            attn_dim=64,
            dropout=0.3,
            num_classes=3,
            return_logits_default=False,
        )
        self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        try:
            with open(self.scaler_path, "rb") as f:
                self.scaler = pickle.load(f)
        except FileNotFoundError:
            # Fallback to per-stock scaling when shared scaler file is missing.
            self.scaler = None

        self.window_size = 60
        self.predict_days = 20

        base_config = default_config if default_config else {
            "entry_up_threshold": 0.55,
            "entry_down_cap": 0.28,
            "entry_margin": 0.18,
            "add_up_threshold": 0.65,
            "exit_down_threshold": 0.55,
        }
        self.default_config = dict(base_config)

        self.stock_configs = {}
        if stock_configs:
            for code, cfg in stock_configs.items():
                norm_code = normalize_stock_code(code)
                self.stock_configs[norm_code] = dict(cfg)

        self.classification_configs = {}
        if classification_configs:
            for group, cfg in classification_configs.items():
                self.classification_configs[str(group)] = dict(cfg)

        self.stock_classification_map = {}
        if stock_classification_map:
            for code, group in stock_classification_map.items():
                norm_code = normalize_stock_code(code)
                self.stock_classification_map[norm_code] = str(group)

        pos_cfg = position_config or {}
        self.initial_buy_ratio = pos_cfg.get("initial_buy_ratio", 0.2)
        self.add_buy_ratio = pos_cfg.get("add_buy_ratio", 0.2)
        self.max_trades_total = pos_cfg.get("max_trades_total", 12)
        self.max_buys_in_window = pos_cfg.get("max_buys_in_window", 3)
        self.recent_buy_lookback_days = pos_cfg.get("recent_buy_lookback_days", 10)
        self.hard_stop_loss = pos_cfg.get("hard_stop_loss", 0.06)
        self.trailing_min_profit = pos_cfg.get("trailing_min_profit", 0.05)
        self.trailing_drawdown = pos_cfg.get("trailing_drawdown", 0.08)
        self.time_stop_days = pos_cfg.get("time_stop_days", 10)
        self.time_stop_band = pos_cfg.get("time_stop_band", 0.02)
        self.partial_take_profit_ratio = pos_cfg.get("partial_take_profit_ratio", 0.3)

        costs = cost_config or {"commission": 0.0003, "tax": 0.001, "slippage": 0.0002}
        self.costs = {
            "commission": costs.get("commission", 0.0003),
            "tax": costs.get("tax", 0.001),
            "slippage": costs.get("slippage", 0.0002),
        }

    # ------------------------------------------------------------------
    def predict(self, features_seq: np.ndarray) -> dict | None:
        if features_seq.shape != (60, 24):
            return None

        X = torch.FloatTensor(features_seq).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(X, return_logits=True)
            probs = torch.softmax(logits, dim=1)[0]

        return {
            "down": probs[0].item(),
            "neutral": probs[1].item(),
            "up": probs[2].item(),
        }

    # ------------------------------------------------------------------
    def get_stock_config(self, stock_code: str) -> dict:
        stock_code = normalize_stock_code(stock_code)

        if stock_code in self.stock_configs:
            return self._merge_with_default(self.stock_configs[stock_code])

        classification = None
        if self.stock_classification_map:
            classification = self.stock_classification_map.get(stock_code)

        if classification and classification in self.classification_configs:
            return self._merge_with_default(self.classification_configs[classification])

        if stock_code.startswith(("688", "300")):
            high_vol_cfg = None
            if "high_volatility" in self.classification_configs:
                high_vol_cfg = self.classification_configs["high_volatility"]
            elif "high_volatility" in self.stock_configs:
                high_vol_cfg = self.stock_configs["high_volatility"]
            if high_vol_cfg:
                return self._merge_with_default(high_vol_cfg)

        return dict(self.default_config)

    # ------------------------------------------------------------------
    def backtest_stock(
        self,
        df: pd.DataFrame,
        stock_code: str,
        initial_capital: float = 10_000_000,
        include_details: bool = False,
    ) -> dict:
        if len(df) < self.window_size + self.predict_days:
            return self._empty_result(stock_code, initial_capital, include_details)

        available_cols = [col for col in FEATURE_COLS if col in df.columns]
        if len(available_cols) < 20:
            return self._empty_result(stock_code, initial_capital, include_details)

        try:
            if self.scaler is not None:
                features_scaled = self.scaler.transform(df[available_cols].values)
            else:
                scaler_temp = StandardScaler()
                features_scaled = scaler_temp.fit_transform(df[available_cols].values)

            features_scaled = np.nan_to_num(features_scaled, nan=0.0)

            if "market" in df.columns:
                market_ma20 = df["market"].rolling(20).mean()
            else:
                market_ma20 = pd.Series([0] * len(df), index=df.index)
        except Exception:
            return self._empty_result(stock_code, initial_capital, include_details)

        cfg = self.get_stock_config(stock_code)
        entry_up_thr = cfg["entry_up_threshold"]
        entry_down_cap = cfg["entry_down_cap"]
        entry_margin = cfg["entry_margin"]
        add_up_thr = cfg["add_up_threshold"]
        exit_down_thr = cfg["exit_down_threshold"]

        cash = initial_capital
        shares = 0
        avg_cost = 0.0
        trades: list[dict] = []
        asset_curve = [initial_capital]

        highest_price_since_entry = 0.0
        days_held = 0

        for i in range(self.window_size, len(df) - self.predict_days):
            current_price = df.iloc[i]["close"]
            current_date = df.index[i]

            is_market_bad = False
            if "market" in df.columns and i > 20:
                if df.iloc[i]["market"] < market_ma20.iloc[i]:
                    is_market_bad = True

            hist_features = features_scaled[i - self.window_size : i]
            probs = self.predict(hist_features)
            if probs is None:
                continue

            p_up = probs["up"]
            p_down = probs["down"]

            if shares > 0:
                days_held += 1
                highest_price_since_entry = max(highest_price_since_entry, current_price)
                unrealized_pnl = (current_price - avg_cost) / avg_cost
                drawdown_from_high = (
                    (current_price - highest_price_since_entry) / highest_price_since_entry
                    if highest_price_since_entry
                    else 0.0
                )

                sell_signal = False
                sell_reason = ""

                if unrealized_pnl < -self.hard_stop_loss:
                    sell_signal = True
                    sell_reason = f"止损 (-{self.hard_stop_loss * 100:.0f}%)"
                elif (
                    unrealized_pnl > self.trailing_min_profit
                    and drawdown_from_high < -self.trailing_drawdown
                ):
                    sell_signal = True
                    sell_reason = "移动止盈"
                elif p_down > exit_down_thr:
                    sell_signal = True
                    sell_reason = "模型看空"
                elif is_market_bad and unrealized_pnl < 0:
                    sell_signal = True
                    sell_reason = "大盘破位风控"
                elif (
                    days_held > self.time_stop_days
                    and -self.time_stop_band < unrealized_pnl < self.time_stop_band
                ):
                    sell_signal = True
                    sell_reason = "时间止损"

                if sell_signal:
                    sell_price = current_price * (
                        1 - self.costs["slippage"] - self.costs["commission"]
                    )
                    proceeds = shares * sell_price
                    tax = proceeds * self.costs["tax"]
                    cash += proceeds - tax

                    realized_pnl = (sell_price - avg_cost) / avg_cost
                    trades.append(
                        {
                            "date": current_date,
                            "type": f"SELL ({sell_reason})",
                            "price": sell_price,
                            "shares": shares,
                            "profit": realized_pnl * 100,
                        }
                    )
                    shares = 0
                    days_held = 0
                    avg_cost = 0.0
                    highest_price_since_entry = 0.0

            if not is_market_bad:
                trades_total = len(trades)
                recent_buys = sum(
                    1
                    for t in trades
                    if "BUY" in t["type"]
                    and (current_date - t["date"]).days < self.recent_buy_lookback_days
                )
                can_trade_now = (
                    trades_total < self.max_trades_total
                    and recent_buys < self.max_buys_in_window
                )

                buy_value = 0.0
                buy_reason = ""

                if can_trade_now:
                    if shares == 0:
                        margin_ok = (p_up - p_down) > entry_margin
                        if p_up > entry_up_thr and p_down < entry_down_cap and margin_ok:
                            buy_value = initial_capital * self.initial_buy_ratio
                            buy_reason = "建仓"
                    elif shares > 0:
                        unrealized_pnl = (current_price - avg_cost) / avg_cost
                        if unrealized_pnl > 0.03 and p_up > add_up_thr:
                            if cash > initial_capital * self.add_buy_ratio:
                                buy_value = initial_capital * self.add_buy_ratio
                                buy_reason = "加仓"

                if buy_value > 0:
                    buy_price = current_price * (
                        1 + self.costs["slippage"] + self.costs["commission"]
                    )
                    new_shares = int(buy_value / buy_price)

                    if new_shares > 0 and cash >= new_shares * buy_price:
                        total_cost_before = shares * avg_cost
                        this_cost = new_shares * buy_price

                        cash -= this_cost
                        shares += new_shares
                        avg_cost = (total_cost_before + this_cost) / shares
                        highest_price_since_entry = max(highest_price_since_entry, current_price)

                        trades.append(
                            {
                                "date": current_date,
                                "type": f"BUY ({buy_reason})",
                                "price": buy_price,
                                "shares": new_shares,
                                "profit": 0,
                            }
                        )

            current_asset = cash + shares * current_price
            asset_curve.append(current_asset)

        if shares > 0:
            final_price = df.iloc[-1]["close"]
            sell_price = final_price * (
                1 - self.costs["slippage"] - self.costs["commission"]
            )
            proceeds = shares * sell_price
            tax = proceeds * self.costs["tax"]
            cash += proceeds - tax

        asset_curve_arr = np.array(asset_curve)
        total_return = (asset_curve_arr[-1] - initial_capital) / initial_capital * 100
        trading_days = len(df)
        annual_return = (
            (1 + total_return / 100) ** (252 / trading_days) - 1 if trading_days > 0 else 0
        )
        annual_return *= 100

        cummax = np.maximum.accumulate(asset_curve_arr)
        max_dd = np.min((asset_curve_arr - cummax) / cummax) * 100

        result = {
            "stock": stock_code,
            "total_return": total_return,
            "annual_return": annual_return,
            "max_drawdown": max_dd,
            "num_trades": len(trades),
            "final_asset": float(asset_curve_arr[-1]),
        }

        if include_details:
            result["trades"] = trades
            result["asset_curve"] = asset_curve

        return result

    # ------------------------------------------------------------------
    def _merge_with_default(self, overrides: dict | None) -> dict:
        merged = dict(self.default_config)
        if overrides:
            merged.update(overrides)
        return merged

    # ------------------------------------------------------------------
    def _empty_result(
        self,
        stock_code: str,
        initial_capital: float,
        include_details: bool,
    ) -> dict:
        result = {
            "stock": stock_code,
            "total_return": 0,
            "annual_return": 0,
            "max_drawdown": 0,
            "num_trades": 0,
            "final_asset": initial_capital,
            "error": True,
        }
        if include_details:
            result["trades"] = []
            result["asset_curve"] = [initial_capital]
        return result

    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_resource_path(path_like: str | Path, base_dir: Path) -> Path:
        path_obj = Path(path_like)
        if path_obj.is_absolute():
            return path_obj

        if path_obj.exists():
            return path_obj

        return base_dir / path_obj