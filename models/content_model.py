import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import sigmoid_kernel, cosine_similarity
import streamlit as st
from utils.data_loader import preprocess_for_content_based


@st.cache_resource
def build_content_model():
    try:
        books_df, tfidf_matrix, indices, tfv = preprocess_for_content_based()

        if books_df.empty or tfidf_matrix is None or indices is None or tfv is None:
            st.error("Failed to preprocess data for content-based filtering")
            return None, None, None, None

        return books_df, tfidf_matrix, indices, tfv
    except Exception as e:
        st.error(f"Error building content model: {e}")
        return None, None, None, None


def get_content_recommendations(book_title, n=10):
    try:
        books_df, tfidf_matrix, indices, _ = build_content_model()

        if books_df is None or tfidf_matrix is None or indices is None:
            st.error("Failed to build content model")
            return pd.DataFrame()

        if book_title not in indices:
            st.error(f"Book '{book_title}' not found in the dataset")
            # Try partial matching
            matching_titles = [
                title for title in indices.index if book_title.lower() in title.lower()
            ]
            if matching_titles:
                book_title = matching_titles[0]
                st.info(f"Using '{book_title}' for recommendations")
            else:
                return pd.DataFrame()

        idx = indices[book_title]

        if isinstance(idx, pd.Series) or isinstance(idx, np.ndarray):
            idx = idx.iloc[0]

        # Calculate similarity scores
        sig = sigmoid_kernel(tfidf_matrix, tfidf_matrix)
        sig_scores = list(enumerate(sig[idx]))

        sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)

        sig_scores = sig_scores[1 : n + 1]

        book_indices = [i[0] for i in sig_scores]

        recommendations = books_df.iloc[book_indices]

        recommendations["similarity_score"] = [score[1] for score in sig_scores]

        return recommendations

    except Exception as e:
        st.error(f"Error getting content recommendations: {e}")
        return pd.DataFrame()


def recommend_from_description(description, n=10):
    try:
        books_df, tfidf_matrix, _, tfv = build_content_model()

        if books_df is None or tfidf_matrix is None or tfv is None:
            st.error("Failed to build content model")
            return pd.DataFrame()

        user_vector = tfv.transform([description])

        similarities = cosine_similarity(user_vector, tfidf_matrix).flatten()

        top_indices = similarities.argsort()[-n:][::-1]

        recommendations = books_df.iloc[top_indices].copy()

        recommendations["similarity_score"] = similarities[top_indices]

        return recommendations

    except Exception as e:
        st.error(f"Error getting recommendations from description: {e}")
        return pd.DataFrame()


def find_similar_books_content(book_title, n=10):
    if not book_title:
        st.warning("Please enter a book title")
        return pd.DataFrame()

    with st.spinner(
        f"Finding books similar to '{book_title}' using content analysis..."
    ):
        recommendations = get_content_recommendations(book_title, n)

        if not recommendations.empty:
            st.success(
                f"Found {len(recommendations)} recommendations for '{book_title}'"
            )

        return recommendations


def find_books_by_description(description, n=10):

    if not description:
        st.warning("Please enter a description")
        return pd.DataFrame()

    with st.spinner("Finding books matching your description..."):
        recommendations = recommend_from_description(description, n)

        if not recommendations.empty:
            st.success(f"Found {len(recommendations)} books matching your description")

        return recommendations
