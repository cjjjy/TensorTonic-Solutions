import numpy as np

def euclidean_distance(x, y):
    """
    计算向量 x 和 y 之间的欧几里得 (L2) 距离。
    必须返回一个 float。
    """
    # 确保输入为 numpy 数组
    x = np.array(x)
    y = np.array(y)
    
    # 按照公式：计算差值 -> 平方 -> 求和 -> 开根号
    distance = np.sqrt(np.sum((x - y)**2))
    
    # 也可以使用 numpy 内置的范数函数，效果相同且更简洁：
    # distance = np.linalg.norm(x - y)
    
    return float(distance)