"""
三分类 GRU 模型（改进版）
- 加深网络：3层GRU，隐层128维
- 加入Attention：自动关注最关键的时间步
- 增加Dropout：防止过拟合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """
    注意力机制层 - 让模型关注序列中最重要的时间步
    
    工作原理：
    - 对GRU的所有输出计算权重
    - 权重高的时间步对最终决策影响大
    - 适合检测"大跌那一天"等关键时刻
    """
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, gru_output):
        """
        Args:
            gru_output: (batch_size, seq_len, hidden_dim)
        
        Returns:
            context: (batch_size, hidden_dim) - 加权平均后的输出
            attention_weights: (batch_size, seq_len) - 注意力权重
        """
        # 计算注意力权重
        attention_scores = self.attention(gru_output)  # (batch_size, seq_len, 1)
        attention_weights = self.softmax(attention_scores.squeeze(-1))  # (batch_size, seq_len)
        
        # 加权求和
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch_size, 1, seq_len)
            gru_output  # (batch_size, seq_len, hidden_dim)
        ).squeeze(1)  # (batch_size, hidden_dim)
        
        return context, attention_weights


class GRUModelTriclass(nn.Module):
    """
    改进的三分类 GRU 模型
    
    架构改进：
    ✓ 加深网络：3层 GRU（从2层），隐层128维（从64维）
    ✓ 加入注意力：自动关注60天中最关键的几天
    ✓ 增加正则化：Dropout从0.2提升到0.4
    
    - 输入: (batch_size, seq_len=60, input_dim=24)
    - 输出: (batch_size, 3) - 三个类别的概率
    """
    
    def __init__(self, input_dim, hidden_dim=128, num_layers=3, dropout=0.4, use_attention=True):
        """
        Args:
            input_dim: 输入特征维度 (通常为24)
            hidden_dim: GRU隐层维度 (改为128，从64)
            num_layers: GRU层数 (改为3，从2)
            dropout: Dropout概率 (改为0.4，从0.2)
            use_attention: 是否使用注意力机制
        """
        super(GRUModelTriclass, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.use_attention = use_attention
        
        # 输入投影层（可选，用于特征预处理）
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # 改进的GRU：更深、更宽
        self.gru = nn.GRU(
            hidden_dim,  # 使用投影后的维度
            hidden_dim, 
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 注意力层
        if use_attention:
            self.attention = AttentionLayer(hidden_dim)
        
        # 分类头：更深的全连接网络，增加正则化
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),  # 增加中间层维度
            nn.ReLU(),
            nn.Dropout(dropout),  # 更强的Dropout
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),  # 额外的Dropout层
            
            nn.Linear(32, 3)  # 输出3个类别
        )
        
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, x, return_logits=False, return_attention=False):
        """
        Args:
            x: (batch_size, seq_len, input_dim)
            return_logits: 是否返回原始logits（用于训练）
            return_attention: 是否返回注意力权重
        
        Returns:
            logits 或 probs: (batch_size, 3)
            (可选) attention_weights: (batch_size, seq_len)
        """
        # 输入投影
        x_proj = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)
        
        # GRU forward
        gru_output, _ = self.gru(x_proj)  # (batch_size, seq_len, hidden_dim)
        
        # 应用注意力或取最后一步
        if self.use_attention:
            out, attention_weights = self.attention(gru_output)
        else:
            out = gru_output[:, -1, :]  # 取最后一个时间步
            attention_weights = None
        
        # 分类头
        logits = self.fc(out)  # (batch_size, 3)
        
        result = logits if return_logits else self.softmax(logits)
        
        if return_attention and self.use_attention:
            return result, attention_weights
        else:
            return result
    
    def predict(self, x):
        """
        预测类别 (用于推理)
        Returns: (batch_size,) 的标签数组，值为 0, 1, 2
        """
        with torch.no_grad():
            if self.use_attention:
                probs, _ = self.forward(x, return_logits=False, return_attention=True)
            else:
                probs = self.forward(x, return_logits=False, return_attention=False)
        
        class_idx = torch.argmax(probs, dim=1)
        return class_idx
    
    def predict_proba(self, x):
        """
        获取各类别的概率
        Returns: 
            probs: (batch_size, 3)
            其中 probs[:, 0] = P(弱势)
                 probs[:, 1] = P(平均)
                 probs[:, 2] = P(强势)
        """
        with torch.no_grad():
            return self.forward(x, return_logits=False)


class ConvGRUAttentionTriclass(nn.Module):
    """
    单股序列版 Conv1D -> GRU -> Attention -> 全连接三分类。
    - 输入: (batch, seq_len, in_feats) 例如 (batch, 60, 24)
    - 输出: (batch, 3) 概率或 logits（可选）
    """

    def __init__(self, in_feats=24, conv_channels=64, conv_k=5,
                 rnn_hidden=128, rnn_layers=1, attn_dim=64,
                 dropout=0.3, num_classes=3, return_logits_default=False):
        super().__init__()
        self.return_logits_default = return_logits_default

        # 时间轴卷积，捕捉局部模式
        self.conv1 = nn.Conv1d(in_channels=in_feats, out_channels=conv_channels,
                               kernel_size=conv_k, padding=conv_k // 2)
        self.bn1 = nn.BatchNorm1d(conv_channels)
        self.conv2 = nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(conv_channels)
        self.dropout = nn.Dropout(dropout)

        # GRU 处理卷积后的序列
        self.gru = nn.GRU(input_size=conv_channels, hidden_size=rnn_hidden,
                          num_layers=rnn_layers, batch_first=True, bidirectional=False)

        # 简单加性注意力
        self.attn_proj = nn.Linear(rnn_hidden, attn_dim)
        self.attn_query = nn.Linear(attn_dim, 1, bias=False)

        # 分类头
        self.fc = nn.Sequential(
            nn.Linear(rnn_hidden, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, return_logits=None, return_attention=False):
        # x: (B, T, F)
        if return_logits is None:
            return_logits = self.return_logits_default

        # Conv1d 期望 (B, C, T)
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # -> (B, T, C)

        gru_out, _ = self.gru(x)  # (B, T, H)

        # 注意力聚合
        attn_hidden = torch.tanh(self.attn_proj(gru_out))       # (B, T, attn_dim)
        attn_scores = self.attn_query(attn_hidden).squeeze(-1)  # (B, T)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, T, 1)
        context = torch.sum(gru_out * attn_weights, dim=1)      # (B, H)

        logits = self.fc(context)  # (B, num_classes)
        out = logits if return_logits else self.softmax(logits)

        if return_attention:
            return out, attn_weights.squeeze(-1)
        return out

    def predict(self, x):
        with torch.no_grad():
            probs = self.forward(x, return_logits=False, return_attention=False)
            return torch.argmax(probs, dim=1)

    def predict_proba(self, x):
        with torch.no_grad():
            return self.forward(x, return_logits=False, return_attention=False)


class FocalLoss(nn.Module):
    """
    Focal Loss - 对困难样本赋予更高权重
    有助于处理类别不平衡问题
    
    参考: https://arxiv.org/abs/1708.02002
    """
    def __init__(self, alpha=None, gamma=2.0, weight=None):
        """
        Args:
            alpha: 类别权重 (list 或 tensor)
            gamma: 专注参数，通常为2
            weight: CrossEntropy的权重
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, reduction='none')
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, num_classes)
            targets: (batch_size,) 目标标签
        """
        ce_loss = self.ce_loss(logits, targets)
        
        # 获取目标类别的概率
        probs = torch.softmax(logits, dim=1)
        p = probs.gather(1, targets.view(-1, 1))
        
        # Focal loss: FL = -alpha * (1-p)^gamma * log(p)
        focal_weight = (1 - p.squeeze()) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        return focal_loss.mean()


class WeightedCrossEntropyLoss(nn.Module):
    """
    带权重的交叉熵损失
    用于处理类别不平衡
    """
    def __init__(self, class_weights, device=None):
        """
        Args:
            class_weights: dict 或 tensor，类别权重
            device: 运算设备 (cuda or cpu)
        """
        super(WeightedCrossEntropyLoss, self).__init__()
        
        self.device = device or torch.device('cpu')
        
        if isinstance(class_weights, torch.Tensor):
            weights = class_weights.to(self.device)
        elif isinstance(class_weights, dict):
            # 支持两种dict格式
            if -1 in class_weights:
                # 旧格式: {-1, 0, 1}
                weights = torch.tensor([
                    class_weights.get(-1, 1.0),
                    class_weights.get(0, 1.0),
                    class_weights.get(1, 1.0)
                ], dtype=torch.float32, device=self.device)
            else:
                # 新格式: {0, 1, 2}
                weights = torch.tensor([
                    class_weights.get(0, 1.0),
                    class_weights.get(1, 1.0),
                    class_weights.get(2, 1.0)
                ], dtype=torch.float32, device=self.device)
        else:
            weights = None
        
        self.ce_loss = nn.CrossEntropyLoss(weight=weights)
    
    def forward(self, logits, targets):
        """
        Args:
            logits: (batch_size, 3)
            targets: (batch_size,) 目标标签，值为 0, 1, 2
        """
        return self.ce_loss(logits, targets)
