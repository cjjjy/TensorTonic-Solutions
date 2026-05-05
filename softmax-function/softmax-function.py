import numpy as np

def softmax(x):
    """
    计算输入 x 的 softmax。
    支持 1D 数组或 2D NumPy 矩阵（按行计算）。
    """
    x = np.array(x)
    
    # 为了数值稳定性，减去每行的最大值 (防止 e^x 溢出)
    # 如果是 2D，使用 keepdims=True 保证维度匹配
    if x.ndim == 1:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    else:
        # 2D 矩阵：按行 (axis=1) 寻找最大值并求和
        max_x = np.max(x, axis=1, keepdims=True)
        e_x = np.exp(x - max_x)
        return e_x / np.sum(e_x, axis=1, keepdims=True)