import requests
import os
from PIL import Image
from io import BytesIO
import streamlit as st
import time
from functools import lru_cache

# a directory for caching images
CACHE_DIR = "assets/image_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

PLACEHOLDER_PATH = "assets/placeholder.png"

@lru_cache(maxsize=100)
def get_book_cover(isbn, size='M'):
    # Check if image is already cached
    cache_path = os.path.join(CACHE_DIR, f"{isbn}_{size}.jpg")
    if os.path.exists(cache_path):
        return cache_path
    
    open_lib_url = f"https://covers.openlibrary.org/b/isbn/{isbn}-{size}.jpg"
    
    try:
        response = requests.head(open_lib_url)
        if response.status_code == 200 and int(response.headers.get('Content-Length', 0)) > 1000:
            # Download and cache the image
            img_response = requests.get(open_lib_url)
            img = Image.open(BytesIO(img_response.content))
            img.save(cache_path)
            return cache_path

    except Exception as e:
        st.error(f"Error fetching image from Open Library: {e}")
    
    try:
        google_books_url = f"https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}"
        response = requests.get(google_books_url)
        data = response.json()
        
        if 'items' in data and 'imageLinks' in data['items'][0]['volumeInfo']:
            image_links = data['items'][0]['volumeInfo']['imageLinks']
            img_url = None
            
            if size == 'S' and 'smallThumbnail' in image_links:
                img_url = image_links['smallThumbnail']
            elif 'thumbnail' in image_links:
                img_url = image_links['thumbnail']
                
            if img_url:
                img_response = requests.get(img_url)
                img = Image.open(BytesIO(img_response.content))
                img.save(cache_path)
                return cache_path
    except Exception as e:
        st.error(f"Error fetching image from Google Books: {e}")
    
    return PLACEHOLDER_PATH

# Pre-fetch and cache multiple book covers
@st.cache_data
def cache_book_covers(isbns, size='M'):
    
    for isbn in isbns:
        # delay to bypass rate limiting
        time.sleep(0.1)
        get_book_cover(isbn, size)


def get_image_for_book(book_data):
    if 'ISBN' in book_data:
        isbn = book_data['ISBN']
        image_path = get_book_cover(isbn)
        if image_path != PLACEHOLDER_PATH:
            return image_path
    
    if 'Book-Title' in book_data:
        title = book_data['Book-Title']
        try:
            search_url = f"https://www.googleapis.com/books/v1/volumes?q=intitle:{title.replace(' ', '+')}"
            response = requests.get(search_url)
            data = response.json()
            
            if 'items' in data and 'imageLinks' in data['items'][0]['volumeInfo']:
                img_url = data['items'][0]['volumeInfo']['imageLinks'].get('thumbnail')
                if img_url:
                    cache_path = os.path.join(CACHE_DIR, f"{title.replace(' ', '_')[:30]}.jpg")
                    
                    img_response = requests.get(img_url)
                    img = Image.open(BytesIO(img_response.content))
                    img.save(cache_path)
                    return cache_path
        except Exception as e:
            st.error(f"Error searching for book image by title: {e}")
    
    return PLACEHOLDER_PATH
