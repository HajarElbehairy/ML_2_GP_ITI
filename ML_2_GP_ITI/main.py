import os
import pandas as pd
import numpy as np
import argparse
from src.preprocessing import download_nltk_resources, full_preprocessing_pipeline
from src.feature_extraction import (create_tfidf_matrix, perform_lda, 
                                   reduce_dimensions_pca, reduce_dimensions_umap)
from src.clustering import (perform_kmeans, perform_hierarchical_clustering,
                           compute_linkage_matrix)
from src.evaluation import (calculate_silhouette_score, find_optimal_clusters_silhouette,
                          calculate_wcss)
from src.visualization import (plot_silhouette_scores, plot_elbow_method,
                            plot_clusters_2d, plot_dendrogram)

def main(args):
    # Create output directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Download required NLTK resources
    download_nltk_resources()
    
    # Load data
    print(f"Loading data from {args.data_path}")
    data = pd.read_csv(args.data_path)
    
    # Preprocess text
    print("Preprocessing text data...")
    data['clean_text'] = data['text'].apply(full_preprocessing_pipeline)
    
    # Feature extraction
    print("Extracting features using TF-IDF...")
    doc_term_matrix, vectorizer = create_tfidf_matrix(data['clean_text'])
    
    # Topic modeling
    print(f"Performing LDA topic modeling with {args.n_topics} topics...")
    lda_model, topic_dist = perform_lda(doc_term_matrix, n_topics=args.n_topics)
    
    # Dimensionality reduction for visualization
    if args.dim_reduction == 'pca':
        print("Reducing dimensions using PCA...")
        reduced_data = reduce_dimensions_pca(topic_dist, n_components=2)
    else:
        print("Reducing dimensions using UMAP...")
        reduced_data = reduce_dimensions_umap(topic_dist, n_components=2)
    
    # Find optimal number of clusters
    print("Finding optimal number of clusters...")
    silhouette_scores, optimal_clusters = find_optimal_clusters_silhouette(
        reduced_data, max_clusters=args.max_clusters
    )
    
    # Plot silhouette scores
    print("Plotting silhouette scores...")
    silhouette_plot_path = os.path.join(args.output_dir, 'silhouette_scores.png')
    plot_silhouette_scores(silhouette_scores, max_clusters=args.max_clusters, 
                          save_path=silhouette_plot_path)
    
    # Calculate WCSS for elbow method
    print("Calculating WCSS for elbow method...")
    wcss = calculate_wcss(reduced_data, max_clusters=args.max_clusters)
    
    # Plot elbow method
    print("Plotting elbow method...")
    elbow_plot_path = os.path.join(args.output_dir, 'elbow_method.png')
    plot_elbow_method(wcss, max_clusters=args.max_clusters, save_path=elbow_plot_path)
    
    # Use the optimal number of clusters if not specified
    n_clusters = args.n_clusters if args.n_clusters > 0 else optimal_clusters
    print(f"Using {n_clusters} clusters for final clustering")
    
    # Perform K-means clustering
    print("Performing K-means clustering...")
    kmeans_model, kmeans_clusters = perform_kmeans(reduced_data, n_clusters=n_clusters)
    
    # Calculate silhouette score
    kmeans_silhouette = calculate_silhouette_score(reduced_data, kmeans_clusters)
    print(f"K-means silhouette score: {kmeans_silhouette:.4f}")
    
    # Plot K-means clusters
    print("Plotting K-means clusters...")
    kmeans_plot_path = os.path.join(args.output_dir, 'kmeans_clusters.png')
    plot_clusters_2d(reduced_data, kmeans_clusters, 
                    title=f"K-means Clustering (Silhouette: {kmeans_silhouette:.4f})",
                    save_path=kmeans_plot_path)
    
    # Perform hierarchical clustering
    print("Performing hierarchical clustering...")
    agg_model, agg_clusters = perform_hierarchical_clustering(reduced_data, n_clusters=n_clusters)
    
    # Calculate silhouette score for hierarchical clustering
    agg_silhouette = calculate_silhouette_score(reduced_data, agg_clusters)
    print(f"Hierarchical clustering silhouette score: {agg_silhouette:.4f}")
    
    # Plot hierarchical clusters
    print("Plotting hierarchical clusters...")
    agg_plot_path = os.path.join(args.output_dir, 'hierarchical_clusters.png')
    plot_clusters_2d(reduced_data, agg_clusters, 
                    title=f"Hierarchical Clustering (Silhouette: {agg_silhouette:.4f})",
                    save_path=agg_plot_path)
    
    # Compute and plot dendrogram
    print("Computing linkage matrix for dendrogram...")
    linkage_matrix = compute_linkage_matrix(reduced_data)
    
    print("Plotting dendrogram...")
    dendrogram_path = os.path.join(args.output_dir, 'dendrogram.png')
    plot_dendrogram(linkage_matrix, save_path=dendrogram_path)
    
    # Save results
    print("Saving results...")
    data['kmeans_cluster'] = kmeans_clusters
    data['hierarchical_cluster'] = agg_clusters
    results_path = os.path.join(args.output_dir, 'clustering_results.csv')
    data.to_csv(results_path, index=False)
    
    print(f"All results saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text Clustering Pipeline")
    parser.add_argument("--data_path", type=str, default="data/people_wiki.csv",
                       help="Path to the input CSV file")
    parser.add_argument("--output_dir", type=str, default="results",
                       help="Directory to save results and visualizations")
    parser.add_argument("--n_topics", type=int, default=10,
                       help="Number of topics for LDA")
    parser.add_argument("--n_clusters", type=int, default=0,
                       help="Number of clusters (0 for auto-detection)")
    parser.add_argument("--max_clusters", type=int, default=15,
                       help="Maximum number of clusters to try")
    parser.add_argument("--dim_reduction", type=str, default="umap", choices=["pca", "umap"],
                       help="Dimensionality reduction method")
    
    args = parser.parse_args()
    main(args)