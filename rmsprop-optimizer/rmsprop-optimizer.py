import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    执行一次 RMSProp 更新步骤。
    """
    # 关键修正：确保输入是 numpy 数组，以便进行元素级运算
    w = np.array(w)
    g = np.array(g)
    s = np.array(s)

    # 步骤 1: 更新平方梯度的运行平均值 (Update Running Average)
    # s_t = beta * s_{t-1} + (1 - beta) * g_t^2
    s_new = beta * s + (1 - beta) * (g ** 2)
    
    # 步骤 2: 更新参数 (Parameter Update)
    # w_t = w_{t-1} - (η / sqrt(s_t + ε)) * g_t
    w_new = w - (lr / (np.sqrt(s_new + eps))) * g
    
    # 如果平台要求返回列表格式，可以使用 .tolist()
    return w_new.tolist(), s_new.tolist()