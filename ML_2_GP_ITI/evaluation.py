from sklearn.metrics import silhouette_score
import numpy as np

def calculate_silhouette_score(data, labels):
    """
    Calculate silhouette score for clustering evaluation.
    
    Args:
        data: Input data
        labels: Cluster labels
        
    Returns:
        float: Silhouette score
    """
    try:
        score = silhouette_score(data, labels)
        return score
    except Exception as e:
        print(f"Error calculating silhouette score: {e}")
        return -1

def find_optimal_clusters_silhouette(data, max_clusters=15, method='kmeans', random_state=42):
    """
    Find optimal number of clusters using silhouette score.
    
    Args:
        data: Input data
        max_clusters (int): Maximum number of clusters to try
        method (str): Clustering method ('kmeans' or other)
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (list of scores, optimal number of clusters)
    """
    silhouette_scores = []
    
    for n_clusters in range(2, max_clusters + 1):
        if method == 'kmeans':
            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        else:
            # Default to KMeans if method not recognized
            from sklearn.cluster import KMeans
            model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            
        labels = model.fit_predict(data)
        
        # Compute silhouette score
        try:
            score = silhouette_score(data, labels)
            silhouette_scores.append(score)
        except Exception:
            silhouette_scores.append(-1)
    
    # Find optimal number of clusters
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    
    return silhouette_scores, optimal_clusters

def calculate_wcss(data, max_clusters=15, random_state=42):
    """
    Calculate Within-Cluster Sum of Squares (WCSS) for elbow method.
    
    Args:
        data: Input data
        max_clusters (int): Maximum number of clusters to try
        random_state (int): Random seed for reproducibility
        
    Returns:
        list: WCSS values for different number of clusters
    """
    from sklearn.cluster import KMeans
    
    wcss = []
    K = range(1, max_clusters + 1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    
    return wcss