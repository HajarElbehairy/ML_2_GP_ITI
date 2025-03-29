from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import linkage

def perform_kmeans(data, n_clusters=3, random_state=42, n_init=10):
    """
    Perform K-means clustering.
    
    Args:
        data: Input data for clustering
        n_clusters (int): Number of clusters
        random_state (int): Random seed for reproducibility
        n_init (int): Number of initializations
        
    Returns:
        tuple: (KMeans model, cluster labels)
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    clusters = kmeans.fit_predict(data)
    return kmeans, clusters

def perform_hierarchical_clustering(data, n_clusters=3, linkage_method='ward'):
    """
    Perform hierarchical/agglomerative clustering.
    
    Args:
        data: Input data for clustering
        n_clusters (int): Number of clusters
        linkage_method (str): Linkage method for hierarchical clustering
        
    Returns:
        tuple: (AgglomerativeClustering model, cluster labels)
    """
    agg_clustering = AgglomerativeClustering(
        n_clusters=n_clusters, 
        linkage=linkage_method,
        compute_full_tree=False
    )
    clusters = agg_clustering.fit_predict(data)
    return agg_clustering, clusters

def compute_linkage_matrix(data, method='ward'):
    """
    Compute linkage matrix for hierarchical clustering.
    
    Args:
        data: Input data
        method (str): Linkage method
        
    Returns:
        numpy.ndarray: Linkage matrix
    """
    return linkage(data, method=method)