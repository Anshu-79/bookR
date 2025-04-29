import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
import streamlit as st

# Define paths to datasets
BOOKS1_PATH = "notebooks/dataset/reviews/BX_Books - 1.csv"
BOOKS2_PATH = "notebooks/dataset/reviews/BX_Books - 2.csv"
RATINGS_PATH = "notebooks/dataset/reviews/BX-Book-Ratings.csv"
CLEAN_BOOKS_PATH = "notebooks/dataset/categorical/books_clean.csv"


@st.cache_data
def load_books_data():
    try:
        books_df1 = pd.read_csv(BOOKS1_PATH, sep=";", encoding="latin-1")
        books_df1 = books_df1[
            [
                "ISBN",
                "Book-Title",
                "Book-Author",
                "Year-Of-Publication",
                "Publisher",
                "Image-URL-M",
            ]
        ]
        books_df2 = pd.read_csv(BOOKS2_PATH, sep=";", encoding="latin-1")
        books_df2 = books_df2[
            [
                "ISBN",
                "Book-Title",
                "Book-Author",
                "Year-Of-Publication",
                "Publisher",
                "Image-URL-M",
            ]
        ]
        
        books_df = pd.concat([books_df1, books_df2], axis=0, ignore_index=True)
        
        return books_df
    except Exception as e:
        st.error(f"Error loading books data: {e}")
        return pd.DataFrame()


@st.cache_data
def load_ratings_data():
    try:
        ratings_df = pd.read_csv(RATINGS_PATH, sep=";", encoding="latin-1")
        return ratings_df
    except Exception as e:
        st.error(f"Error loading ratings data: {e}")
        return pd.DataFrame()


@st.cache_data
def load_clean_books_data():
    try:
        books_df = pd.read_csv(CLEAN_BOOKS_PATH)
        return books_df
    except Exception as e:
        st.error(f"Error loading clean books data: {e}")
        try:
            books_df = load_books_data()
            return books_df
        except:
            return pd.DataFrame()


@st.cache_data
def create_book_matrix(min_ratings=100):
    try:
        ratings_df = load_ratings_data()
        books_df = load_books_data()

        df = pd.merge(ratings_df, books_df, on="ISBN")

        book_ratings_count = df.groupby("ISBN")["Book-Rating"].count()
        popular_books = book_ratings_count[book_ratings_count >= min_ratings].index

        df_popular = df[df["ISBN"].isin(popular_books)]

        book_matrix = df_popular.pivot_table(
            index="User-ID", columns="ISBN", values="Book-Rating"
        )

        return book_matrix

    except Exception as e:
        st.error(f"Error creating book matrix: {e}")
        return pd.DataFrame()


@st.cache_data
def preprocess_for_content_based():
    try:
        books_df = load_clean_books_data()

        books_df.rename(columns={"isbn10": "ISBN"}, inplace=True)

        # If clean books dataset is empty, use the regular books dataset
        if books_df.empty:
            books_df = load_books_data()
            # Create a description field if it doesn't exist
            if "description" not in books_df.columns:
                books_df["description"] = (
                    books_df["Book-Title"] + " by " + books_df["Book-Author"]
                )

        books_df = calculate_weighted_hybrid(books_df)

        # Fill missing descriptions
        books_df["description"] = books_df["description"].fillna("")

        if "title" in books_df.columns:
            content = books_df["title"] + ": " + books_df["description"]
        else:
            content = books_df["Book-Title"] + ": " + books_df["description"]

        tfv = TfidfVectorizer(
            min_df=3,
            max_features=None,
            strip_accents="unicode",
            analyzer="word",
            token_pattern=r"\w{1,}",
            ngram_range=(1, 3),
            stop_words="english",
        )

        tfidf_matrix = tfv.fit_transform(content)

        if "title" in books_df.columns:
            indices = pd.Series(
                books_df.index, index=books_df["title"]
            ).drop_duplicates()
        else:
            indices = pd.Series(
                books_df.index, index=books_df["Book-Title"]
            ).drop_duplicates()

        return books_df, tfidf_matrix, indices, tfv

    except Exception as e:
        st.error(f"Error preprocessing for content-based filtering: {e}")
        return pd.DataFrame(), None, None, None


@st.cache_data
def get_book_titles_starting_with(prefix: str):
    try:
        books_df = load_clean_books_data()
        titles = books_df["title"].unique()
        filtered_titles = [
            title for title in titles if title.lower().startswith(prefix.lower())
        ]
        return sorted(filtered_titles)
    except Exception as e:
        st.error(f"Error getting book titles: {e}")
        return []


def calculate_weighted_hybrid(books):
    top500_fraction = (len(books) - 500) / len(books)

    R = books["average_rating"]
    v = books["ratings_count"]
    C = books["average_rating"].mean()
    m = books["ratings_count"].quantile(top500_fraction)

    books["weighted_avg"] = (R * v + C * m) / (v + m)
    
    scaling = MinMaxScaler()
    
    books_scaled_df = scaling.fit_transform(books[['weighted_avg', 'ratings_count']])

    books[['normalized_weight_avg', 'normalized_popularity']] = pd.DataFrame(books_scaled_df)
    
    books['score'] = books['normalized_weight_avg'] * 0.5 + books['normalized_popularity'] * 0.5
    
    return books
