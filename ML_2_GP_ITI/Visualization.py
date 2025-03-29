import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram
import numpy as np

def plot_silhouette_scores(scores, max_clusters=15, save_path=None):
    """
    Plot silhouette scores for different number of clusters.
    
    Args:
        scores (list): List of silhouette scores
        max_clusters (int): Maximum number of clusters evaluated
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(10, 5))
    plt.plot(range(2, max_clusters + 1), scores, marker='o')
    plt.title('Silhouette Score vs Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_elbow_method(wcss, max_clusters=15, save_path=None):
    """
    Plot WCSS values for elbow method.
    
    Args:
        wcss (list): List of WCSS values
        max_clusters (int): Maximum number of clusters evaluated
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='-')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for Optimal k')
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_clusters_2d(data_2d, labels, title="Cluster Visualization", 
                    x_label="Dimension 1", y_label="Dimension 2", 
                    colormap='viridis', alpha=0.7, save_path=None):
    """
    Plot 2D data points colored by cluster labels.
    
    Args:
        data_2d: 2D data points
        labels: Cluster labels
        title (str): Plot title
        x_label (str): X-axis label
        y_label (str): Y-axis label
        colormap (str): Matplotlib colormap
        alpha (float): Transparency of points
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, 
                        cmap=colormap, alpha=alpha)
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()

def plot_dendrogram(linkage_matrix, truncate_mode="level", p=10, save_path=None):
    """
    Plot dendrogram for hierarchical clustering.
    
    Args:
        linkage_matrix: Linkage matrix from hierarchical clustering
        truncate_mode (str): Mode for truncating the dendrogram
        p (int): Parameter for truncation
        save_path (str, optional): Path to save the figure
    """
    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix, truncate_mode=truncate_mode, p=p)
    plt.title("Dendrogram for Hierarchical Clustering")
    plt.xlabel("Samples")
    plt.ylabel("Distance")
    
    if save_path:
        plt.savefig(save_path)
    
    plt.show()