import streamlit as st
import pandas as pd
import os

# Import utility modules
from utils.data_loader import get_book_titles_starting_with
from utils.ui_components import (
    apply_custom_css,
    create_header,
    create_recommendation_grid,
    create_model_selection_buttons,
    create_search_box,
    create_description_search_box,
    create_footer,
    create_divider,
)
from utils.image_fetcher import cache_book_covers

# Import recommendation models
from models.knn_model import find_similar_books_knn
from models.correlation_model import find_similar_books_correlation
from models.content_model import find_similar_books_content, find_books_by_description

# Create assets directory if it doesn't exist
os.makedirs("assets", exist_ok=True)
os.makedirs("assets/image_cache", exist_ok=True)

# Set page configuration
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Apply custom CSS
apply_custom_css("assets/custom.css")

# Create header
create_header()

# Initialize session state for storing recommendations
if "recommendations" not in st.session_state:
    st.session_state.recommendations = pd.DataFrame()

if "active_model" not in st.session_state:
    st.session_state.active_model = None

# Create search box
book_title = create_search_box(get_book_titles_starting_with)

# Create model selection buttons
knn_button, correlation_button, content_button = create_model_selection_buttons()

# Display active model
if st.session_state.active_model:
    model_names = {
        "knn": "K-Nearest Neighbors",
        "correlation": "Pearson Correlation",
        "content": "Content-Based",
        "description": "Description-Based",
    }
    st.markdown(
        f"<p style='color: #3581B8; font-weight: bold;'>Active Model: {model_names.get(st.session_state.active_model, 'Unknown')}</p>",
        unsafe_allow_html=True,
    )


create_divider()

# Create description search box
description, description_search_button = create_description_search_box()

# Handle model button clicks
if knn_button:
    st.session_state.active_model = "knn"
    if book_title:
        st.session_state.recommendations = find_similar_books_knn(book_title)

if correlation_button:
    st.session_state.active_model = "correlation"
    if book_title:
        st.session_state.recommendations = find_similar_books_correlation(book_title)

if content_button:
    st.session_state.active_model = "content"
    if book_title:
        st.session_state.recommendations = find_similar_books_content(book_title)

# Handle description search
if description_search_button and description:
    st.session_state.recommendations = find_books_by_description(description)
    st.session_state.active_model = "description"


# Display recommendations
if not st.session_state.recommendations.empty:
    st.markdown("<h2>Recommended Books</h2>", unsafe_allow_html=True)

    # Pre-cache book covers for better performance
    if "ISBN" in st.session_state.recommendations.columns:
        cache_book_covers(st.session_state.recommendations["ISBN"].tolist())

    # Display recommendations in a grid
    create_recommendation_grid(st.session_state.recommendations, cols=2)


# Create footer
create_footer()
