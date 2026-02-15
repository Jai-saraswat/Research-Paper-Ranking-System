# Research Paper Ranking System 📚🔍

A scalable, intelligent research paper search and ranking system that combines multiple information retrieval techniques to deliver highly relevant search results. The system leverages hybrid semantic embeddings, TF-IDF lexical matching, and metadata features to rank academic papers effectively.

## 📋 Table of Contents
- [Overview](#overview)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Current Implementation](#current-implementation)
- [Future Enhancements](#future-enhancements)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Technologies & Techniques](#technologies--techniques)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)

## 🎯 Overview

The Research Paper Ranking System is designed to help researchers, students, and academics quickly find relevant research papers from a large corpus. Given a user query, the system returns the top-ranked papers based on semantic similarity, lexical matching, and various metadata signals such as citations, author count, and references.

**Key Features:**
- **Hybrid Search**: Combines semantic understanding with traditional keyword matching
- **Multi-feature Scoring**: Incorporates citations, references, and author information
- **Scalable Design**: Efficiently processes 1M+ papers using vectorized operations
- **Fast Query Processing**: Pre-computed embeddings and sparse matrices for real-time ranking

## 🔧 How It Works

The system uses a multi-stage ranking pipeline:

### 1. **Query Processing**
   - User enters a natural language query
   - Query is converted into both semantic embeddings and TF-IDF vectors

### 2. **Similarity Computation**
   - **Semantic Similarity**: Compares query embeddings with pre-computed paper embeddings (abstract and title)
   - **Lexical Similarity**: Computes TF-IDF cosine similarity for keyword matching

### 3. **Feature Engineering**
   - Extracts metadata features: citation count, reference count, author count
   - Normalizes features using logarithmic scaling where appropriate

### 4. **Scoring & Ranking**
   - Combines all features using a weighted scoring formula
   - Current weights:
     - Abstract embedding similarity: 35%
     - Title embedding similarity: 25%
     - Combined TF-IDF score: 20%
     - Citation count (log-scaled): 10%
     - Reference count: 5%
     - Author count: 5%

### 5. **Result Delivery**
   - Returns top 20 ranked papers with title, venue, and year

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          USER QUERY                              │
│                     (Natural Language Text)                      │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    QUERY PROCESSING LAYER                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────┐      ┌──────────────────────┐        │
│  │  Semantic Embedding  │      │   TF-IDF Vectorizer  │        │
│  │   (MiniLM-L6-v2)    │      │   (Sklearn)          │        │
│  └──────────────────────┘      └──────────────────────┘        │
└────────────────────────┬────────────────┬───────────────────────┘
                         │                │
                         ▼                ▼
┌─────────────────────────────────────────────────────────────────┐
│                  SIMILARITY COMPUTATION LAYER                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │         PRE-COMPUTED PAPER REPRESENTATIONS               │  │
│  ├──────────────────────────────────────────────────────────┤  │
│  │  • Abstract Embeddings (NumPy)                           │  │
│  │  • Title Embeddings (NumPy)                              │  │
│  │  • Abstract TF-IDF Matrix (Sparse)                       │  │
│  │  • Title TF-IDF Matrix (Sparse)                          │  │
│  │  • Metadata Features (Parquet)                           │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Cosine Similarity Computation:                                 │
│  • Query ↔ Abstract Embeddings                                  │
│  • Query ↔ Title Embeddings                                     │
│  • Query ↔ Abstract TF-IDF                                      │
│  • Query ↔ Title TF-IDF                                         │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    SCORING & RANKING LAYER                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Final Score = Weighted Combination:                             │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ 0.35 × Abstract_Embedding_Similarity                       │ │
│  │ 0.25 × Title_Embedding_Similarity                          │ │
│  │ 0.20 × Combined_TF-IDF_Score                               │ │
│  │ 0.10 × log(Citation_Count + 1)                             │ │
│  │ 0.05 × Reference_Count                                     │ │
│  │ 0.05 × Author_Count                                        │ │
│  └────────────────────────────────────────────────────────────┘ │
│                                                                  │
│  Sort by Final Score (Descending)                               │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                      TOP 20 RANKED PAPERS                        │
│                  (Title, Venue, Year, etc.)                      │
└─────────────────────────────────────────────────────────────────┘
```

## 📊 Current Implementation

The current version uses a **Direct Scoring Formula** approach:
- Pre-computed embeddings and TF-IDF matrices are loaded at startup
- Query-time scoring combines multiple signals with fixed weights
- No machine learning model training required for ranking
- Deterministic and interpretable scoring mechanism

### Direct Scoring Formula Details:
```python
final_score = (
    0.35 × abstract_embedding_similarity +
    0.25 × title_embedding_similarity +
    0.20 × combined_tfidf_score +
    0.10 × log(n_citation + 1) +
    0.05 × ref_count +
    0.05 × author_count
)
```

This approach provides:
- ✅ Fast query-time performance
- ✅ Interpretable results
- ✅ No training data requirements
- ✅ Consistent behavior

## 🚀 Future Enhancements

The system is designed with extensibility in mind. Planned enhancements include:

### 1. **Learning-to-Rank (LTR) Model**
   - Replace fixed weights with learned parameters
   - Train on user click data and relevance judgments
   - Optimize ranking metrics (NDCG, MRR, MAP)
   - Techniques: LambdaMART, RankNet, or LTR-specific models

### 2. **Document Clustering**
   - Group similar papers into topical clusters
   - Enable cluster-based navigation and exploration
   - Improve diversity in search results
   - Techniques: K-Means, HDBSCAN on embeddings

### 3. **Advanced Retrieval**
   - Approximate Nearest Neighbor (ANN) search using FAISS or Annoy
   - Two-stage retrieval: fast candidate generation + precise ranking
   - Further scalability improvements for 10M+ papers

## 📁 Dataset

This project uses the **Research Papers Dataset** from Kaggle:

🔗 **Dataset Link**: [https://www.kaggle.com/datasets/nechbamohammed/research-papers-dataset](https://www.kaggle.com/datasets/nechbamohammed/research-papers-dataset)

**Dataset Contents:**
- Research paper titles
- Abstracts
- Authors
- Publication venues
- Publication years
- Citation counts
- References
- And more metadata...

The dataset should be downloaded and processed to generate the required assets (embeddings, TF-IDF matrices, and feature files) stored in the `Core/` directory.

## 📂 Project Structure

```
Research-Paper-Ranking-System/
│
├── main.py                          # Main entry point for the ranking system
│
├── Core/                            # Pre-computed assets directory
│   ├── Ranking.ipynb               # Jupyter notebook for data preparation
│   ├── abstract_embeddings.npy     # Pre-computed abstract embeddings
│   ├── title_embeddings.npy        # Pre-computed title embeddings
│   ├── X_abstract.npz              # Sparse TF-IDF matrix for abstracts
│   ├── X_title.npz                 # Sparse TF-IDF matrix for titles
│   ├── abstract_vec.pkl            # Trained abstract TF-IDF vectorizer
│   ├── title_vec.pkl               # Trained title TF-IDF vectorizer
│   ├── original_data_parquet       # Original dataset in Parquet format
│   └── training_features_parquet   # Pre-computed features for all papers
│
├── requirements.txt                 # Python dependencies
├── .gitignore                       # Git ignore rules
└── README.md                        # This file
```

### Key Files:

- **`main.py`**: The core ranking pipeline that loads assets, processes queries, computes similarities, and returns ranked results.
- **`Core/Ranking.ipynb`**: Jupyter notebook for data preprocessing, generating embeddings, TF-IDF matrices, and feature extraction.
- **Pre-computed Assets**: Binary files (`.npy`, `.npz`, `.pkl`, `.parquet`) containing processed data for fast query-time operations.

## 🛠️ Technologies & Techniques

### **Libraries & Frameworks**
- **NumPy**: Efficient numerical computations and array operations
- **Pandas**: Data manipulation and analysis
- **SciPy**: Sparse matrix operations for efficient TF-IDF storage
- **Scikit-learn**: TF-IDF vectorization and cosine similarity
- **Sentence Transformers**: Semantic embeddings using pre-trained models
- **Joblib**: Model serialization and deserialization
- **PyArrow/Fastparquet**: Efficient Parquet file I/O

### **Techniques**
1. **Semantic Embeddings**:
   - Model: `sentence-transformers/all-MiniLM-L6-v2`
   - Captures semantic meaning of text
   - Dense vector representation (384 dimensions)

2. **TF-IDF (Term Frequency-Inverse Document Frequency)**:
   - Traditional lexical matching
   - Sparse matrix representation for memory efficiency
   - Captures keyword importance

3. **Cosine Similarity**:
   - Measures similarity between query and document vectors
   - Efficient vectorized computation

4. **Feature Engineering**:
   - Logarithmic scaling for citation counts (handles skewed distributions)
   - Normalization of metadata features
   - Multi-signal fusion

5. **Hybrid Ranking**:
   - Combines semantic and lexical signals
   - Weighted aggregation of multiple features
   - Balances relevance and popularity

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for faster embedding generation)
- At least 8GB RAM (16GB+ recommended for large datasets)

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Jai-saraswat/Research-Paper-Ranking-System.git
   cd Research-Paper-Ranking-System
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**:
   - Visit [Kaggle Dataset](https://www.kaggle.com/datasets/nechbamohammed/research-papers-dataset)
   - Download the research papers dataset
   - Place it in an accessible location

5. **Generate pre-computed assets**:
   - Open `Core/Ranking.ipynb` in Jupyter Notebook
   - Follow the notebook to:
     - Load and preprocess the dataset
     - Generate embeddings for abstracts and titles
     - Create TF-IDF matrices
     - Extract and save metadata features
   - The notebook will save all assets in the `Core/` directory

### Project Initialization

After generating the assets, your `Core/` directory should contain:
- `abstract_embeddings.npy`
- `title_embeddings.npy`
- `X_abstract.npz`
- `X_title.npz`
- `abstract_vec.pkl`
- `title_vec.pkl`
- `original_data_parquet`
- `training_features_parquet`

## 💻 Usage

### Running the Ranking System

1. **Start the system**:
   ```bash
   python main.py
   ```

2. **Enter your query**:
   ```
   Ask for Research Papers: machine learning applications in healthcare
   ```

3. **View results**:
   The system will display the top 20 ranked papers with their titles, venues, and publication years.

### Example Session

```bash
$ python main.py
Loading Assets...
Assets Loaded.
Ask for Research Papers: deep learning for image classification
Processing Query...
Query Completed.

Top Results:

                                                title                    venue  year
ResNet: Deep Residual Learning for Image Recognition                      CVPR  2015
ImageNet Classification with Deep Convolutional Neural Networks           NIPS  2012
...
```

## 🤝 Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding features, improving documentation, or suggesting enhancements, your help is appreciated.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**:
   - Write clean, documented code
   - Follow existing code style
   - Add tests if applicable
4. **Commit your changes**:
   ```bash
   git commit -m "Add: description of your changes"
   ```
5. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Open a Pull Request**:
   - Provide a clear description of your changes
   - Reference any related issues

### Contribution Ideas

- 🎯 Implement Learning-to-Rank models
- 📊 Add document clustering functionality
- 🔍 Integrate ANN-based retrieval (FAISS, Annoy)
- 🎨 Create a web interface (Flask/FastAPI + React)
- 📈 Add evaluation metrics and benchmarking
- 📝 Improve documentation and tutorials
- 🐛 Fix bugs and optimize performance
- ✅ Add unit tests and integration tests

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Help others learn and grow

## 📄 License

This project is open source and available for anyone to use, modify, and contribute to.

## 📧 Contact

For questions, suggestions, or discussions, feel free to open an issue or reach out through GitHub.

---

**Happy Researching! 📚✨**
