import numpy as np

def adagrad_step(w, g, G, lr=0.01, eps=1e-8):
    """
    执行一次 AdaGrad 更新步骤。
    """
    # 关键修正：确保输入是 numpy 数组
    w = np.array(w)
    g = np.array(g)
    G = np.array(G)
    
    # 步骤 1: 累积平方梯度 (Accumulate Squared Gradients)
    # G_t = G_{t-1} + g_t^2
    G_new = G + (g ** 2)
    
    # 步骤 2: 参数更新 (Parameter Update)
    # w_t = w_{t-1} - (lr / sqrt(G_t + eps)) * g_t
    w_new = w - (lr / (np.sqrt(G_new + eps))) * g
    
    # 建议：如果平台严格要求返回列表，可以使用 .tolist()，
    # 但通常 TensorTonic 支持返回 ndarray。
    return w_new, G_new