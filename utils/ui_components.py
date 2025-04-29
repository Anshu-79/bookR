import streamlit as st
from streamlit_searchbox import st_searchbox
import pandas as pd
from utils.image_fetcher import get_image_for_book
import base64
import math


def apply_custom_css(css_file):
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def create_header():
    st.markdown(
        "<h1 style='text-align: center;'>📚 Book Recommendation System</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align: center;'>Discover your next favorite book using advanced recommendation algorithms</p>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr>", unsafe_allow_html=True)


def create_book_card(book, key=None):

    if isinstance(book, pd.Series):
        book = book.to_dict()

    title = book.get("Book-Title", book.get("title", "Unknown Title"))
    author = book.get("Book-Author", book.get("authors", "Unknown Author"))
    rating = book.get(
        "Book-Rating", book.get("average_rating", book.get("rating", "N/A"))
    )
    weighted_avg = book.get("score", book.get("weighted_avg", "N/A"))

    rating = f"{rating:.2f}" if isinstance(rating, float) else rating
    weighted_avg = (
        f"{weighted_avg:.2f}"
        if isinstance(weighted_avg, float) and not math.isnan(weighted_avg)
        else "N/A"
    )

    isbn = book.get("ISBN", "")

    image_path = get_image_for_book(book)

    col1, col2 = st.columns([1, 3])

    with col1:
        st.image(image_path, width=150)

    with col2:
        st.markdown(f"<h4>{title}</h4>", unsafe_allow_html=True)
        st.markdown(f"<p class='author'>by {author}</p>", unsafe_allow_html=True)

        st.markdown(
            f"<p class='rating'>Average Rating: {rating}</p>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<p class='rating'>Weighted Hybrid Score: {weighted_avg}</p>",
            unsafe_allow_html=True,
        )
        if isbn:
            st.markdown(f"<p>ISBN: {isbn}</p>", unsafe_allow_html=True)


def create_recommendation_grid(books, cols=2):

    if isinstance(books, pd.DataFrame):
        books_list = books.to_dict("records")
    else:
        books_list = books

    for i in range(0, len(books_list), cols):
        columns = st.columns(cols)
        for j in range(cols):
            if i + j < len(books_list):
                with columns[j]:
                    with st.container():
                        st.markdown("<div class='book-card'>", unsafe_allow_html=True)
                        create_book_card(books_list[i + j], key=f"book_{i}_{j}")
                        st.markdown("</div>", unsafe_allow_html=True)


def create_model_selection_buttons():
    st.markdown(
        "<p style='font-weight: bold;'>Pick your preferred recommendation model:</p>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        knn_button = st.button("KNN Model", key="knn_button", use_container_width=True)

    with col2:
        correlation_button = st.button(
            "Pearson Correlation", key="correlation_button", use_container_width=True
        )

    with col3:
        content_button = st.button(
            "Content-Based", key="content_button", use_container_width=True
        )

    return knn_button, correlation_button, content_button


def create_search_box(get_book_titles):
    book_title = st_searchbox(
        get_book_titles,
        placeholder="Enter a book title",
        key="book_title",
    )

    return book_title


def create_divider():
    st.markdown(
        """
    <div style="display: flex; align-items: center; text-align: center; margin: 2rem 0;">
        <hr>
        <span style="padding: 0 1rem; font-weight: bold; font-size: 1.2rem;">OR</span>
        <hr>
    </div>
    """,
        unsafe_allow_html=True,
    )


def create_description_search_box():
    st.markdown(
        "<p style='font-weight: bold;'>Describe a book you'd like to read:</p>",
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([4, 1])

    with col1:
        description = st.text_area("", key="description", height=68)

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        description_search_button = st.button(
            "Find Books", key="description_search_button", use_container_width=True
        )

    return description, description_search_button


def create_loading_placeholder():
    return st.empty()


def show_loading_animation(placeholder):
    placeholder.markdown(
        """
        <div style="display: flex; justify-content: center; margin: 2rem 0;">
            <div style="border: 8px solid #1E1E1E; border-top: 8px solid #FF521B; border-radius: 50%; width: 50px; height: 50px; animation: spin 1s linear infinite;"></div>
        </div>
        <style>
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def clear_loading(placeholder):
    placeholder.empty()


def create_footer():
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class="footer">
            <p>Anshu79's Book Recommendation System | Built with Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
