import pandas as pd
import numpy as np
import streamlit as st
from utils.data_loader import load_books_data, load_ratings_data, calculate_weighted_hybrid


# Create the correlation matrix for book recommendations
@st.cache_resource
def build_correlation_matrix(popularity_threshold=100):

    try:
        ratings_df = load_ratings_data()
        books_df = load_books_data()

        df = pd.merge(ratings_df, books_df, on="ISBN")
        
        ratings_with_count = pd.DataFrame(df.groupby("ISBN")["Book-Rating"].mean())
        ratings_with_count["ratings_count"] = pd.DataFrame(
            df.groupby("ISBN")["Book-Rating"].count()
        )

        ratings_with_count = ratings_with_count[
            ratings_with_count["ratings_count"] > popularity_threshold
        ]

        # Calculate Weighted Hybrid Rating
        ratings_with_count.rename(columns={'Book-Rating': 'average_rating'}, inplace=True)
        ratings_with_count = calculate_weighted_hybrid(ratings_with_count)
        
        books_df = books_df.merge(ratings_with_count, on='ISBN')

        df = df[df["ISBN"].isin(ratings_with_count.index)]

        ratings_with_count.rename(columns={'average_rating': 'Book-Rating'}, inplace=True)
        book_matrix = df.pivot_table(
            index="User-ID", columns="ISBN", values="Book-Rating"
        )

        ratings_with_count = ratings_with_count.merge(
            df[["ISBN", "Book-Title"]].drop_duplicates(),
            left_index=True,
            right_on="ISBN",
        )

        return book_matrix, ratings_with_count, books_df
    except Exception as e:
        st.error(f"Error building correlation matrix: {e}")
        return None, None, None


def get_correlation_recommendations(book_title, n=10, min_ratings=75):
    try:
        book_matrix, ratings_df, books_df = build_correlation_matrix()

        if book_matrix is None or ratings_df is None or books_df is None:
            st.error("Failed to build correlation matrix")
            return pd.DataFrame()

        book_isbn = None
        for isbn, title in zip(ratings_df["ISBN"], ratings_df["Book-Title"]):
            if title.lower() == book_title.lower():
                book_isbn = isbn
                break

        if not book_isbn:
            for isbn, title in zip(ratings_df["ISBN"], ratings_df["Book-Title"]):
                if book_title.lower() in title.lower():
                    book_isbn = isbn
                    book_title = title
                    st.info(f"Using '{book_title}' for recommendations")
                    break

        if not book_isbn:
            st.error(
                f"Book '{book_title}' not found in the dataset or doesn't have enough ratings"
            )
            return pd.DataFrame()

        book_user_ratings = book_matrix[book_isbn]

        similar_to_book = book_matrix.corrwith(book_user_ratings)

        corr_book = pd.DataFrame(similar_to_book, columns=["Correlation"])
        corr_book.dropna(inplace=True)

        corr_book = corr_book.merge(
            books_df[["ISBN", "Book-Title"]].drop_duplicates(),
            left_index=True,
            right_on="ISBN",
        )

        corr_book = corr_book.join(ratings_df["ratings_count"])

        recommendations = corr_book.sort_values("Correlation", ascending=False)

        recommendations = recommendations[recommendations["Book-Title"] != book_title]

        top_recommendations = recommendations.head(n)

        detailed_recommendations = []
        for _, row in top_recommendations.iterrows():
            book_details = books_df[books_df["ISBN"] == row["ISBN"]].iloc[0].to_dict()
            book_details["Correlation"] = row["Correlation"]
            book_details["ratings_count"] = row["ratings_count"]
            detailed_recommendations.append(book_details)

        return pd.DataFrame(detailed_recommendations)

    except Exception as e:
        st.error(f"Error getting correlation recommendations: {e}")
        return pd.DataFrame()


def find_similar_books_correlation(book_title, n=10):
    if not book_title:
        st.warning("Please enter a book title")
        return pd.DataFrame()

    with st.spinner(
        f"Finding books similar to '{book_title}' using Pearson correlation..."
    ):
        recommendations = get_correlation_recommendations(book_title, n)

        if not recommendations.empty:
            st.success(
                f"Found {len(recommendations)} recommendations for '{book_title}'"
            )

        return recommendations
