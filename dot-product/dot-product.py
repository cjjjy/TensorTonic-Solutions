

import numpy as np

def dot_product(x, y):
    """
    计算两个 1D 数组 x 和 y 的点积。
    必须返回一个浮点数。
    """
    # 确保输入为 numpy 数组以支持高效向量化运算
    x = np.array(x)
    y = np.array(y)
    
    # 方式 A：直接使用 numpy 内置函数（推荐，性能最高）
    result = np.dot(x, y)
    
    # 方式 B：手动实现代数定义
    # result = np.sum(x * y)
    
    return float(result)