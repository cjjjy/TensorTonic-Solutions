import numpy as np

def td_value_update(V, s, r, s_next, alpha, gamma):
    """
    计算并返回更新后的价值函数 V_new。
    """
    # 确保 V 是 numpy 数组，方便后续操作（虽然输入通常已经是了）
    V_new = np.array(V, dtype=float)
    
    # 步骤 1: 计算 TD 误差 (TD Error)
    # delta = r + gamma * V(s_next) - V(s)
    delta = r + gamma * V_new[s_next] - V_new[s]
    
    # 步骤 2: 更新当前状态 s 的价值
    # V(s) = V(s) + alpha * delta
    V_new[s] = V_new[s] + alpha * delta
    
    return V_new