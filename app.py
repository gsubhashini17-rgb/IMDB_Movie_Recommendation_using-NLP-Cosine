import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

download_nltk_data()

# Page config
st.set_page_config(
    page_title="IMDb Movie Recommender",
    page_icon="🎬",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #F5C518;
        text-align: center;
        margin-bottom: 1rem;
    }
    .movie-card {
        background-color: #1E1E1E;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 5px solid #F5C518;
    }
    .movie-title {
        font-size: 1.3rem;
        color: #F5C518;
        font-weight: bold;
    }
    .similarity-badge {
        background-color: #F5C518;
        color: #000000;
        padding: 3px 10px;
        border-radius: 15px;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

class MovieRecommender:
    def __init__(self):
        """Load the pre-trained recommender"""
        try:
            self.df = pd.read_csv('processed_movies.csv')
            self.similarity_matrix = np.load('similarity_matrix.npy')
            with open('tfidf_vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
        except FileNotFoundError:
            st.error("❌ Data files not found! Please ensure all files are present.")
            st.stop()
    
    def preprocess_input(self, text):
        """Preprocess user input"""
        if not text:
            return ""
        
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        
        try:
            tokens = word_tokenize(text)
            processed = [
                self.lemmatizer.lemmatize(token) 
                for token in tokens 
                if token not in self.stop_words and len(token) > 2
            ]
            return ' '.join(processed)
        except:
            return text
    
    def recommend_by_storyline(self, storyline, top_n=5):
        """Get recommendations by storyline"""
        processed = self.preprocess_input(storyline)
        
        if not processed:
            return pd.DataFrame()
        
        input_vector = self.vectorizer.transform([processed])
        similarities = cosine_similarity(
            input_vector, 
            self.vectorizer.transform(self.df['processed_storyline'])
        ).flatten()
        
        top_indices = similarities.argsort()[-top_n:][::-1]
        recommendations = self.df.iloc[top_indices].copy()
        recommendations['similarity_score'] = similarities[top_indices]
        
        return recommendations[['movie_name', 'year', 'rating', 'storyline', 'similarity_score']]
    
    def recommend_by_title(self, title, top_n=5):
        """Get recommendations by movie title"""
        mask = self.df['movie_name'].str.contains(title, case=False, na=False)
        matches = self.df[mask]
        
        if len(matches) == 0:
            return None, f"Movie '{title}' not found"
        
        movie_idx = matches.index[0]
        movie_name = matches.iloc[0]['movie_name']
        
        movie_similarities = self.similarity_matrix[movie_idx]
        similar_indices = movie_similarities.argsort()[-top_n-1:][::-1][1:top_n+1]
        
        recommendations = self.df.iloc[similar_indices].copy()
        recommendations['similarity_score'] = movie_similarities[similar_indices]
        
        return recommendations, movie_name

# Initialize recommender
@st.cache_resource
def load_recommender():
    return MovieRecommender()

recommender = load_recommender()

# Header
st.markdown('<h1 class="main-header">🎬 IMDb Movie Recommendation System</h1>', 
            unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/6/69/IMDB_Logo_2016.svg", 
             use_column_width=True)
    
    st.markdown("## 📊 Dataset Stats")
    st.metric("Total Movies", len(recommender.df))
    st.metric("Years Range", f"{recommender.df['year'].min()} - {recommender.df['year'].max()}")
    st.metric("Avg Rating", f"{recommender.df['rating'].mean():.2f}")

# Main content
tab1, tab2 = st.tabs(["🔍 Recommend by Storyline", "🎯 Recommend by Movie Title"])

with tab1:
    st.markdown("### Enter a movie storyline")
    
    user_input = st.text_area(
        "Describe the movie plot:",
        height=150,
        placeholder="Type or paste a movie storyline here..."
    )
    
    if st.button("🎬 Get Recommendations", type="primary"):
        if user_input:
            with st.spinner("🔍 Finding similar movies..."):
                recommendations = recommender.recommend_by_storyline(user_input, top_n=5)
            
            if not recommendations.empty:
                st.markdown("---")
                st.markdown("## 🎯 Top 5 Recommendations")
                
                for idx, row in recommendations.iterrows():
                    st.markdown(f"""
                    <div class="movie-card">
                        <div style="display: flex; justify-content: space-between;">
                            <span class="movie-title">🎥 {row['movie_name']} ({int(row['year'])}) ⭐ {row['rating']}</span>
                            <span class="similarity-badge">{row['similarity_score']:.1%} match</span>
                        </div>
                        <div style="color: #DDD; margin-top: 10px;">
                            📖 {row['storyline']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("No recommendations found.")
        else:
            st.warning("Please enter a storyline.")

with tab2:
    st.markdown("### Enter a movie title")
    
    movie_title = st.text_input("Movie title:", placeholder="e.g., Inception")
    
    if st.button("Find Similar Movies", type="primary"):
        if movie_title:
            recommendations, found_title = recommender.recommend_by_title(movie_title, top_n=5)
            
            if isinstance(recommendations, pd.DataFrame):
                st.markdown(f"### Movies similar to **{found_title}**")
                
                for idx, row in recommendations.iterrows():
                    st.markdown(f"""
                    <div class="movie-card">
                        <div style="display: flex; justify-content: space-between;">
                            <span class="movie-title">🎥 {row['movie_name']} ({int(row['year'])}) ⭐ {row['rating']}</span>
                            <span class="similarity-badge">{row['similarity_score']:.1%} match</span>
                        </div>
                        <div style="color: #DDD; margin-top: 10px;">
                            📖 {row['storyline']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.error(found_title)
        else:
            st.warning("Please enter a movie title.")