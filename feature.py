import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from glob import glob

# ========================================
# 配置路径
# ========================================
DATA_DIR = "stock_data"
DAY_DIR = os.path.join(DATA_DIR, "day")
DAY_LONG_DIR = os.path.join(DATA_DIR, "day_long")
WEEK_DIR = os.path.join(DATA_DIR, "week")
MONTH_DIR = os.path.join(DATA_DIR, "month")
FEATURE_DIR = "features"

os.makedirs(FEATURE_DIR, exist_ok=True)

# ========================================
# 需要处理的股票代码列表 (91只)
# ========================================
STOCK_CODES = {
    "000858","600519","002304","000568","603369","603589","603198","603919","603986","688981",
    "002371","600703","603501","688187","688008","300661","300223","300782","002049","300373",
    "300346","300567","300458","002812","300014","002460","300450","300750","300014","002466",
    "603659","002594","601633","600104","000625","601238","002708","600900","003816","601985",
    "600011","600023","000993","300359","002261","600661","002315","603877","002563","002291",
    "002425","002569","000157","000425","600031","601100","002097","002202","601615","300443",
    "002531","603606","601012","300274","002459","603806","688599","300118","002242","603486",
    "002508","002032","603355","600547","601899","600489","002155","600311","600030","601995",
    "601688","600837","000776","002736","601066","600999","601318","601628","601601","601336",
    "601319"
}

# ========================================
# 特征构建函数
# ========================================
def compute_task1_features(df_day, df_long, df_week=None, df_month=None):
    """
    构建任务一的特征库：
    1. 日频技术指标
    2. 相对强度（与市场 market 对比）
    3. 周/月周期特征
    4. 长周期特征（day_long）
    5. 数据清洗（填补缺失）
    """
    df = df_long.copy()
    df = df.sort_values("day").reset_index(drop=True)

    # -------------------------------
    # 1. 基础价格/收益/量能
    # -------------------------------
    df["return"] = df["close"].pct_change()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    df["range"] = df["high"] - df["low"]
    df["high_low_ratio"] = df["high"] / df["low"]
    df["close_open_ratio"] = df["close"] / df["open"]
    df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3
    df["volume_change"] = df["volume"].pct_change()
    df["amount"] = df["volume"] * df["close"]  # 生成近似成交额
    df["amount_change"] = df["amount"].pct_change()

    # -------------------------------
    # 2. 日频技术指标
    # -------------------------------
    df["MA5"] = ta.sma(df["close"], 5)
    df["MA10"] = ta.sma(df["close"], 10)
    df["MA20"] = ta.sma(df["close"], 20)
    df["MA60"] = ta.sma(df["close"], 60)

    df["price_MA5_ratio"] = df["close"] / df["MA5"]
    df["price_MA20_ratio"] = df["close"] / df["MA20"]

    try:
        macd = ta.macd(df["close"])
        df["MACD"] = macd["MACD_12_26_9"]
        df["MACD_signal"] = macd["MACDs_12_26_9"]
        df["MACD_hist"] = macd["MACDh_12_26_9"]
    except:
        df["MACD"] = 0
        df["MACD_signal"] = 0
        df["MACD_hist"] = 0

    df["RSI14"] = ta.rsi(df["close"], 14)
    bb = ta.bbands(df["close"])
    df["BBL"], df["BBM"], df["BBU"] = bb.iloc[:,0], bb.iloc[:,1], bb.iloc[:,2]
    df["BB_width"] = (df["BBU"] - df["BBL"]) / df["BBM"]
    df["BB_zscore"] = (df["close"] - df["BBM"]) / (df["BBU"] - df["BBL"])*4

    # -------------------------------
    # 3. 相对市场强度
    # -------------------------------
    df["market_return"] = df["market"].pct_change()
    df["relative_return"] = df["return"] - df["market_return"]

    df["beta_60"] = df["return"].rolling(60).cov(df["market_return"]) / \
                    df["market_return"].rolling(60).var()
    df["corr_60"] = df["return"].rolling(60).corr(df["market_return"])
    df["cum_relative_20"] = df["relative_return"].rolling(20).sum()
    df["cum_relative_60"] = df["relative_return"].rolling(60).sum()

    # -------------------------------
    # 4. 长周期特征
    # -------------------------------
    # if df_long is not None:
    # df_long = df_long.sort_values("day").reset_index(drop=True)

    # 先计算日收益率
    df["return"] = df["close"].pct_change()

    # 长周期动量
    df["momentum_120"] = df["close"] / df["close"].shift(120) - 1
    df["momentum_250"] = df["close"] / df["close"].shift(250) - 1

    # 长周期波动率
    df["volatility_120"] = df["return"].rolling(120).std()

    # # 合并到日频
    # df = df.merge(
    #     df_long[["day", "momentum_120", "momentum_250", "volatility_120"]],
    #     on="day", how="left"
    # )

    # -------------------------------
    # 5. 周/月特征
    # -------------------------------
    df_temp = df.set_index("day")

    if df_week is not None:
        df_week = df_week.set_index("day").sort_index()
        df_week["week_return"] = df_week["close"].pct_change()
        df_week["week_volatility"] = df_week["week_return"].rolling(4).std()
        df_week[["week_return", "week_volatility"]] = df_week[["week_return", "week_volatility"]].shift(1)
        df = df.merge(df_week[["week_return", "week_volatility"]],
                      left_on="day", right_index=True, how="left")

    if df_month is not None:
        df_month = df_month.set_index("day").sort_index()
        df_month["month_return"] = df_month["close"].pct_change()
        df_month["month_volatility"] = df_month["month_return"].rolling(3).std()
        df_month[["month_return", "month_volatility"]] = df_month[["month_return", "month_volatility"]].shift(1)
        df = df.merge(df_month[["month_return", "month_volatility"]],
                      left_on="day", right_index=True, how="left")

    # -------------------------------
    # 6. 量价指标
    # -------------------------------
    df["vol_ma_5"] = df["volume"].rolling(5).mean()
    df["vol_ma_20"] = df["volume"].rolling(20).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma_20"]
    df["OBV"] = (np.sign(df["close"].diff()) * df["volume"]).fillna(0).cumsum()
    df["momentum_10"] = df["close"] / df["close"].shift(10) - 1
    df["momentum_20"] = df["close"] / df["close"].shift(20) - 1
    df["momentum_60"] = df["close"] / df["close"].shift(60) - 1

    # -------------------------------
    # 7. 去极值 & 缺失值
    # -------------------------------
    numeric_cols = df.select_dtypes("number").columns
    for col in numeric_cols:
        p1, p99 = df[col].quantile(0.01), df[col].quantile(0.99)
        df[col] = df[col].clip(p1, p99)

    df = df.fillna(method="ffill").fillna(0)
    return df

# ========================================
# 批量处理所有股票
# ========================================
def process_all_stocks():
    stock_files = glob(os.path.join(DAY_DIR, "*.csv"))
    all_features = []
    successful_stocks = []
    failed_stocks = []

    for file_path in stock_files:
        stock_code = os.path.basename(file_path).replace(".csv", "")
        
        # 只处理在STOCK_CODES列表中的股票
        if stock_code not in STOCK_CODES:
            continue
        
        try:
            print(f"Processing stock {stock_code} ...")

            df_day = pd.read_csv(file_path, parse_dates=["day"])
            df_long_path = os.path.join(DAY_LONG_DIR, f"{stock_code}.csv")
            df_week_path = os.path.join(WEEK_DIR, f"{stock_code}.csv")
            df_month_path = os.path.join(MONTH_DIR, f"{stock_code}.csv")

            df_long = pd.read_csv(df_long_path, parse_dates=["day"]) if os.path.exists(df_long_path) else None
            df_week = pd.read_csv(df_week_path, parse_dates=["day"]) if os.path.exists(df_week_path) else None
            df_month = pd.read_csv(df_month_path, parse_dates=["day"]) if os.path.exists(df_month_path) else None

            df_features = compute_task1_features(df_day, df_long, df_week, df_month)

            # 保存单个股票特征
            df_features.to_csv(os.path.join(FEATURE_DIR, f"{stock_code}_features.csv"), index=False)
            successful_stocks.append(stock_code)
            all_features.append(df_features)
        
        except Exception as e:
            print(f"  ✗ 处理失败: {e}")
            failed_stocks.append((stock_code, str(e)))

    # 合并所有股票特征库
    if all_features:
        df_all = pd.concat(all_features, ignore_index=True)
        df_all.to_csv(os.path.join(FEATURE_DIR, "all_stocks_features.parquet"), index=False)
    
    # 输出汇总信息
    print("\n" + "=" * 100)
    print("特征提取完成")
    print("=" * 100)
    print(f"\n✓ 成功处理: {len(successful_stocks)} 只股票")
    print(f"✗ 处理失败: {len(failed_stocks)} 只股票")
    print(f"📊 缺失数据: {len(STOCK_CODES) - len(successful_stocks) - len(failed_stocks)} 只")
    
    if successful_stocks:
        print(f"\n✓ 已生成的CSV文件 ({len(successful_stocks)} 只):")
        for i, code in enumerate(sorted(successful_stocks), 1):
            print(f"  {i:2d}. {code}_features.csv")
    
    if failed_stocks:
        print(f"\n✗ 处理失败的股票:")
        for code, error in failed_stocks:
            print(f"  {code}: {error}")
    
    missing_stocks = STOCK_CODES - set(successful_stocks) - set([c for c, _ in failed_stocks])
    if missing_stocks:
        print(f"\n❌ 缺失原始数据的股票 ({len(missing_stocks)} 只):")
        for code in sorted(missing_stocks):
            print(f"  {code}")
    
    print(f"\n所有特征文件已保存到: {FEATURE_DIR}/")


# ========================================
# 主程序
# ========================================
if __name__ == "__main__":
    process_all_stocks()