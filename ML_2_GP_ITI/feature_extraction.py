from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, PCA
import umap.umap_ as umap
import numpy as np

def create_tfidf_matrix(texts, **kwargs):
    """
    Convert texts to TF-IDF matrix.
    
    Args:
        texts (list): List of preprocessed text documents
        **kwargs: Additional arguments for TfidfVectorizer
        
    Returns:
        tuple: (TF-IDF matrix, vectorizer)
    """
    vectorizer = TfidfVectorizer(**kwargs)
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer

def perform_lda(doc_term_matrix, n_topics=10, random_state=42):
    """
    Perform Latent Dirichlet Allocation topic modeling.
    
    Args:
        doc_term_matrix: Document-term matrix
        n_topics (int): Number of topics
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (LDA model, document-topic distribution)
    """
    lda_model = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=random_state,
        learning_method='online'
    )
    topic_dist = lda_model.fit_transform(doc_term_matrix)
    return lda_model, topic_dist

def reduce_dimensions_pca(data, n_components=10, random_state=42):
    """
    Reduce dimensionality using PCA.
    
    Args:
        data: High-dimensional data
        n_components (int): Number of components to keep
        random_state (int): Random seed for reproducibility
        
    Returns:
        numpy.ndarray: Reduced data
    """
    reducer = PCA(n_components=n_components, random_state=random_state)
    reduced_data = reducer.fit_transform(data)
    return reduced_data

def reduce_dimensions_umap(data, n_components=2, random_state=42):
    """
    Reduce dimensionality using UMAP.
    
    Args:
        data: High-dimensional data
        n_components (int): Number of components to keep
        random_state (int): Random seed for reproducibility
        
    Returns:
        numpy.ndarray: Reduced data
    """
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    reduced_data = reducer.fit_transform(data)
    return reduced_data