##########################################################
# RESEARCH PAPER RANKING SYSTEM #########################
# Scalable Query-Time Ranking Pipeline
##########################################################

import joblib
from scipy import sparse
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from sentence_transformers import SentenceTransformer

##########################################################
# LOAD ASSETS ############################################
##########################################################

print("Loading Assets...")

abstract_embeddings = np.load(r"Core/abstract_embeddings.npy")
title_embeddings = np.load(r"Core/title_embeddings.npy")

original_df = pd.read_parquet(r"Core/original_data_parquet")
base_features_df = pd.read_parquet(r"Core/training_features_parquet")

abstract_tfidf_matrix = sparse.load_npz(r"Core/X_abstract.npz")
title_tfidf_matrix = sparse.load_npz(r"Core/X_title.npz")

abstract_vectorizer = joblib.load(r"Core/abstract_vec.pkl")
title_vectorizer = joblib.load(r"Core/title_vec.pkl")

model = SentenceTransformer(
    'sentence-transformers/all-MiniLM-L6-v2',
    device='cuda'
)

print("Assets Loaded.")

##########################################################
# SAFE COLUMN ACCESS #####################################
##########################################################

def safe_col(df, name):
    return df[name] if name in df.columns else 0

##########################################################
# QUERY INPUT ############################################
##########################################################

def ask_query():
    return input("Ask for Research Papers: ").strip().lower()

##########################################################
# EMBEDDINGS #############################################
##########################################################

def get_query_embedding(query):
    return model.encode([query], convert_to_numpy=True)

##########################################################
# TFIDF ##################################################
##########################################################

def get_query_tfidf(query):
    return (
        abstract_vectorizer.transform([query]),
        title_vectorizer.transform([query])
    )

##########################################################
# SIMILARITIES ###########################################
##########################################################

def embedding_similarity(q_emb):
    abs_sim = cosine_similarity(q_emb, abstract_embeddings).flatten()
    title_sim = cosine_similarity(q_emb, title_embeddings).flatten()
    return abs_sim, title_sim

def tfidf_similarity(q_abs_vec, q_title_vec):
    abs_scores = cosine_similarity(q_abs_vec, abstract_tfidf_matrix).flatten()
    title_scores = cosine_similarity(q_title_vec, title_tfidf_matrix).flatten()
    return abs_scores, title_scores

##########################################################
# MAIN PIPELINE ##########################################
##########################################################

def run_query_pipeline(query):

    print("Processing Query...")

    # Copy base features
    temp_df = base_features_df.copy()

    # Query vectors
    q_emb = get_query_embedding(query)
    q_abs_vec, q_title_vec = get_query_tfidf(query)

    # Similarities
    abs_emb_sim, title_emb_sim = embedding_similarity(q_emb)
    abs_tfidf_sim, title_tfidf_sim = tfidf_similarity(q_abs_vec, q_title_vec)

    # Add query features
    temp_df['abstract_embedding_similarity'] = abs_emb_sim
    temp_df['title_embedding_similarity'] = title_emb_sim
    temp_df['abstract_tfidf_score'] = abs_tfidf_sim
    temp_df['title_tfidf_score'] = title_tfidf_sim

    temp_df['combined_tfidf_score'] = (
        0.6 * title_tfidf_sim + 0.4 * abs_tfidf_sim
    )

    ######################################################
    # FINAL SCORE ########################################
    ######################################################

    temp_df['final_score'] = (
        0.35 * temp_df['abstract_embedding_similarity'] +
        0.25 * temp_df['title_embedding_similarity'] +
        0.20 * temp_df['combined_tfidf_score'] +
        0.10 * np.log1p(safe_col(temp_df, 'n_citation')) +
        0.05 * safe_col(temp_df, 'ref_count') +
        0.05 * safe_col(temp_df, 'author_count')
    )

    ######################################################
    # SORT ################################################
    ######################################################

    ranked_idx = temp_df['final_score'].values.argsort()[::-1]

    results = original_df.iloc[ranked_idx].head(20)

    print("Query Completed.")
    return results

##########################################################
# ENTRY ##################################################
##########################################################

if __name__ == "__main__":
    query = ask_query()
    results = run_query_pipeline(query)

    print("\nTop Results:\n")
    print(results[['title', 'venue', 'year']])
