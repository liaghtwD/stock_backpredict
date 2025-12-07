from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


# ================= 随机森林模型 =================
def create_rf_model(n_estimators=100, max_depth=10, min_samples_leaf=20, random_state=42):
    """
    创建随机森林分类器

    Args:
        n_estimators: 树的数量
        max_depth: 树的最大深度（防止过拟合）
        min_samples_leaf: 叶子节点最小样本数
        random_state: 随机种子

    Returns:
        RandomForestClassifier模型
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1
    )


# ================= 逻辑回归模型 =================
def create_lr_model(C=1.0, penalty='l2', max_iter=1000, random_state=42):
    """
    创建逻辑回归分类器（用于短期策略）

    Args:
        C: 正则化强度的倒数，越小表示正则化越强
        penalty: 正则化类型 ('l1', 'l2', 'elasticnet', 'none')
        max_iter: 最大迭代次数
        random_state: 随机种子

    Returns:
        LogisticRegression模型
    """
    return LogisticRegression(
        C=C,
        penalty=penalty,
        max_iter=max_iter,
        random_state=random_state,
        class_weight='balanced',
        solver='saga' if penalty == 'elasticnet' or penalty == 'l1' else 'lbfgs'
    )