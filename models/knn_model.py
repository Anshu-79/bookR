import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import streamlit as st
from utils.data_loader import load_books_data, load_ratings_data, calculate_weighted_hybrid


# Create and train the KNN model
@st.cache_resource
def build_knn_model(popularity_threshold=100):
    try:
        ratings_df = load_ratings_data()
        books_df = load_books_data()

        df = pd.merge(ratings_df, books_df, on="ISBN")

        book_num_ratings = (
            df.groupby(by=["Book-Title"])["Book-Rating"]
            .count()
            .reset_index()
            .rename(columns={"Book-Rating": "ratings_count"})[
                ["Book-Title", "ratings_count"]
            ]
        )

        df = df.merge(book_num_ratings, on="Book-Title", how="left")

        rating_popular_books = df[
            df["ratings_count"] >= popularity_threshold
        ]

        # Calculate Weighted Hybrid Rating
        book_avg_rating = (
            df.groupby(by=["Book-Title"])["Book-Rating"]
            .mean()
            .reset_index()
            .rename(columns={"Book-Rating": "average_rating"})[
                ["Book-Title", "average_rating"]
            ]
        )

        books_df = books_df.merge(book_num_ratings, on='Book-Title', how='left')        
        books_df = books_df.merge(book_avg_rating, on='Book-Title', how='left')
        
        books_df = calculate_weighted_hybrid(books_df)

        book_features_df = rating_popular_books.pivot_table(
            index="Book-Title", columns="User-ID", values="Book-Rating"
        ).fillna(0)

        book_features_df_matrix = csr_matrix(book_features_df.values)

        model_knn = NearestNeighbors(metric="cosine", algorithm="brute")
        model_knn.fit(book_features_df_matrix)

        return model_knn, book_features_df, books_df

    except Exception as e:
        st.error(f"Error building KNN model: {e}")
        return None, None, None


def get_knn_recommendations(book_title, n=10):
    try:
        model_knn, book_features_df, books_df = build_knn_model()

        if model_knn is None or book_features_df is None or books_df is None:
            st.error("Failed to build KNN model")
            return pd.DataFrame()

        if book_title not in book_features_df.index:
            st.error(
                f"Book '{book_title}' not found in the dataset or doesn't have enough ratings"
            )

            similar_titles = [
                title
                for title in book_features_df.index
                if book_title.lower() in title.lower()
            ]
            if similar_titles:
                st.info(f"Did you mean one of these? {', '.join(similar_titles[:5])}")
            return pd.DataFrame()

        book_idx = book_features_df.index.get_loc(book_title)

        distances, indices = model_knn.kneighbors(
            book_features_df.iloc[book_idx, :].values.reshape(1, -1),
            n_neighbors=n + 1,  # +1 because the book itself will be included
        )

        recommendations = []
        for i in range(1, len(distances.flatten())):
            book_title_rec = book_features_df.index[indices.flatten()[i]]
            distance = distances.flatten()[i]

            book_details = (
                books_df[books_df["Book-Title"] == book_title_rec].iloc[0].to_dict()
            )
            book_details["similarity_score"] = (
                1 - distance
            )  # Convert distance to similarity score

            recommendations.append(book_details)

        return pd.DataFrame(recommendations)
    except Exception as e:
        st.error(f"Error getting KNN recommendations: {e}")
        return pd.DataFrame()


def find_similar_books_knn(book_title, n=10):
    if not book_title:
        st.warning("Please enter a book title")
        return pd.DataFrame()

    with st.spinner(f"Finding books similar to '{book_title}' using KNN..."):
        recommendations = get_knn_recommendations(book_title, n)

        if not recommendations.empty:
            st.success(
                f"Found {len(recommendations)} recommendations for '{book_title}'"
            )

        return recommendations
