from sklearn.metrics import average_precision_score
import numpy as np

# Recall@K
def recall_at_k(recommended, relevant, k):
    """
    Compute Recall@K: Proportion of relevant items in top K recommended.
    recommended: List of recommended items (indices or IDs).
    relevant: Set of relevant items (indices or IDs).
    k: Top K number of items considered for the recommendation.
    """
    if len(recommended) == 0:
        return 0.0
    relevant_at_k = set(recommended[:k]).intersection(relevant)
    return len(relevant_at_k) / min(k, len(relevant))

# MAP@K
def mean_average_precision_at_k(recommended, relevant, k):
    """
    Compute Mean Average Precision at K (MAP@K).
    recommended: List of recommended items (indices or IDs).
    relevant: Set of relevant items (indices or IDs).
    k: Top K number of items considered for the recommendation.
    """
    average_precision = 0.0
    num_relevant_items = 0
    
    for i in range(min(k, len(recommended))):
        if recommended[i] in relevant:
            num_relevant_items += 1
            average_precision += num_relevant_items / (i + 1)
    
    if num_relevant_items == 0:
        return 0.0
    return average_precision / min(k, len(relevant))

# Example of how to use these functions:
if __name__ == "__main__":
    recommended_items = [1, 3, 4, 7, 8]  # IDs of the recommended items
    relevant_items = {1, 4, 7}           # Set of relevant item IDs (ground truth)
    k = 3

    recall = recall_at_k(recommended_items, relevant_items, k)
    mapk = mean_average_precision_at_k(recommended_items, relevant_items, k)
    
    print(f"Recall@{k}: {recall}")
    print(f"MAP@{k}: {mapk}")
