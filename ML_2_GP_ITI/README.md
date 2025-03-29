# Text Clustering Project

This project implements a text clustering pipeline using natural language processing (NLP) techniques, topic modeling, and various clustering algorithms. The goal is to discover natural groupings within text data.

## Project Structure

```
├── data/                  # Folder for datasets (raw and preprocessed)
├── src/
│   ├── preprocessing.py    # Code for data cleaning and preprocessing
│   ├── feature_extraction.py  # Code for vectorization and embedding
│   ├── clustering.py       # Code for implementing clustering models
│   ├── evaluation.py       # Code for computing clustering metrics
│   ├── visualization.py    # Code for plotting results
│   ├── main.py             # Main script to run the project pipeline
├── notebooks/              # Jupyter notebooks for exploratory data analysis
├── results/                # Folder to save cluster results and visualizations
├── requirements.txt        # List of required Python libraries
├── README.md               # This documentation file
```

## Features

- Text preprocessing: cleaning, tokenization, stopword removal, lemmatization
- Feature extraction: TF-IDF vectorization and Latent Dirichlet Allocation (LDA) topic modeling
- Dimensionality reduction: PCA and UMAP
- Clustering algorithms:
  - K-means
  - Hierarchical (Agglomerative) clustering
- Cluster evaluation: Silhouette scores and elbow method
- Visualization: cluster plots, silhouette analysis, dendrogram, and more

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/text_clustering_project.git
   cd text_clustering_project
   ```

2. Create a virtual environment (optional):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

## Usage

### Running the pipeline

```bash
python src/main.py --data_path data/people_wiki.csv --output_dir results
```

### Command-line arguments

- `--data_path`: Path to the input CSV file (default: `data/people_wiki.csv`)
- `--output_dir`: Directory to save results and visualizations (default: `results`)
- `--n_topics`: Number of topics for LDA (default: 10)
- `--n_clusters`: Number of clusters (default: 0 for auto-detection)
- `--max_clusters`: Maximum number of clusters to try (default: 15)
- `--dim_reduction`: Dimensionality reduction method (choices: `pca`, `umap`; default: `umap`)

## Example

```bash
# Run with default parameters
python src/main.py

# Run with custom parameters
python src/main.py --data_path data/custom_data.csv --n_topics 15 --n_clusters 5 --dim_reduction pca
```

## Results

The pipeline generates:
- Cluster visualizations
- Silhouette and elbow method plots for optimal cluster selection
- Dendrogram for hierarchical clustering
- CSV file with cluster assignments

## License

This project is licensed under the MIT License - see the LICENSE file for details.