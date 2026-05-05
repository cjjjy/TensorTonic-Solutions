import numpy as np

def entropy_node(y):
    """
    使用数值稳定的方法计算单个节点的熵。
    """
    # 1. 统计每个类别的频率
    labels, counts = np.unique(y, return_counts=True)
    
    # 2. 计算各类别所占比例 p_i
    probs = counts / len(y)
    
    # 3. 计算熵：只计算 p > 0 的项，避免 log(0) 报错并消除精度干扰造成的负值
    # 使用布尔索引过滤掉概率为 0 的项（虽然 np.unique 不产生 0 概率，但这是最佳实践）
    safe_probs = probs[probs > 0]
    entropy = -np.sum(safe_probs * np.log2(safe_probs))
    
    # 4. 强制截断微小负值（双重保险）
    return max(0.0, float(entropy))