def precision_recall_at_k(recommended, relevant, k):
    """
    计算推荐列表的 precision@k 和 recall@k。
    """
    # 1. 取出前 k 个推荐项 (top-k)
    top_k = recommended[:k]
    
    # 2. 将列表转换为集合，利用交集计算“命中的相关项”
    set_top_k = set(top_k)
    set_relevant = set(relevant)
    
    # 计算交集：top-k ∩ relevant
    hits = len(set_top_k.intersection(set_relevant))
    
    # 3. 按照公式计算 Precision@K
    precision = hits / k if k > 0 else 0.0
    
    # 4. 按照公式计算 Recall@K
    recall = hits / len(relevant) if len(relevant) > 0 else 0.0
    
    # 关键修正：返回列表而不是元组
    return [precision, recall]