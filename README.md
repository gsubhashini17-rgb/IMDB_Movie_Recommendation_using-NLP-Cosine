# 🎬 IMDb Movie Recommendation System Using Natural Language Processing

[![Streamlit
App](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit)](https://imdbmovierecommendationusing-nlp-cosine-pwtksz7t5pzkhcdx3yqzly.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange?logo=scikit-learn)](https://scikit-learn.org/)
[![GitHub
Repo](https://img.shields.io/badge/GitHub-Repo-black?logo=github)](https://github.com/yourusername/imdb_movie_recommendation_using-nlp-cosine)

## 📋 Project Overview

An intelligent movie recommendation system that suggests movies based on
storyline similarity using Natural Language Processing (NLP) and Machine
Learning techniques. Users can input a movie plot or description, and
the system recommends the top 5 most similar movies from a dataset of
1000 IMDb movies.

**Live Demo:**\
https://imdbmovierecommendationusing-nlp-cosine-pwtksz7t5pzkhcdx3yqzly.streamlit.app/

------------------------------------------------------------------------

## ✨ Features

-   🔍 Recommend movies by storyline
-   🎯 Recommend movies by movie title
-   📊 Real-time similarity scores
-   📈 Interactive visualizations
-   ⭐ Movie details (year, rating, storyline)
-   📱 Responsive design

------------------------------------------------------------------------

## 🛠 Technology Stack

### Core Technologies

-   Python 3.8+
-   Streamlit
-   scikit-learn
-   NLTK
-   Pandas
-   NumPy
-   Plotly

### Key Libraries

    pandas==2.0.3
    numpy==1.24.3
    scikit-learn==1.3.0
    nltk==3.8.1
    streamlit==1.28.0
    plotly==5.17.0

------------------------------------------------------------------------

## 🏗 System Architecture

User Input (Storyline)\
→ Text Cleaning\
→ Tokenization & Lemmatization\
→ TF-IDF Vectorization\
→ Cosine Similarity\
→ Top 5 Movie Recommendations

------------------------------------------------------------------------

## 🔄 How It Works

### Step 1: Text Preprocessing

-   Convert text to lowercase
-   Remove punctuation
-   Tokenization
-   Stopword removal
-   Lemmatization

Example:

Input:\
"A young wizard discovers magical powers"

Output:\
"young wizard discover magical power"

### Step 2: TF‑IDF Vectorization

Converts text to numerical vectors using:

-   Term Frequency (TF)
-   Inverse Document Frequency (IDF)

### Step 3: Cosine Similarity

Formula:

cos(θ) = (A·B) / (\|\|A\|\| × \|\|B\|\|)

Range: - 0 → completely different - 1 → identical

### Step 4: Recommendation Generation

-   Compare input with all movies
-   Sort by similarity score
-   Return top 5 matches

------------------------------------------------------------------------

## 📊 Dataset

Source: Kaggle IMDb dataset

Columns: - name -- Movie title - description -- Plot summary - year --
Release year - rating -- IMDb rating - duration -- Movie length -
metascore -- Critic score

Dataset Stats: - Total Movies: 1000 - Years Range: 1920 -- 2024 -
Average Rating: 8.2 - Top Rated: The Shawshank Redemption (9.3)

------------------------------------------------------------------------

## 💻 Installation & Setup

### Prerequisites

-   Python 3.8+
-   pip
-   Git (optional)

### Steps

Clone repository

    git clone https://github.com/yourusername/imdb_movie_recommendation_using-nlp-cosine.git
    cd imdb_movie_recommendation_using-nlp-cosine

Install dependencies

    pip install -r requirements.txt

Download NLTK data

    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

Run application

    streamlit run app.py

Open browser

    http://localhost:8501

------------------------------------------------------------------------

## 📖 Usage Guide

### Method 1: Recommend by Storyline

1.  Enter a movie storyline
2.  Click **Get Recommendations**
3.  View top 5 similar movies

Example: "A young wizard discovers magical powers and attends a school
of magic."

### Method 2: Recommend by Movie Title

1.  Enter movie title
2.  Click **Find Similar Movies**
3.  Explore similar recommendations

------------------------------------------------------------------------

## 📁 Project Structure

    imdb-movie-recommendation/
    │
    ├── app.py
    ├── requirements.txt
    ├── README.md
    │
    ├── data/
    │   ├── imdb_kaggle.csv
    │   └── processed_movies.csv
    │
    ├── models/
    │   ├── tfidf_vectorizer.pkl
    │   └── similarity_matrix.npy
    │
    └── notebooks/
        ├── 01_data_preprocessing.ipynb
        ├── 02_model_training.ipynb
        └── 03_evaluation.ipynb

------------------------------------------------------------------------

## 📐 Methodology

1.  Data Collection -- Kaggle IMDb dataset\
2.  Data Preprocessing -- cleaning, tokenization, stopword removal,
    lemmatization\
3.  Feature Extraction -- TF‑IDF vectorization\
4.  Similarity Calculation -- cosine similarity matrix\
5.  Recommendation Generation -- return top 5 matches

------------------------------------------------------------------------

## 📈 Results & Insights

Example Recommendations:

  Input                      Top Recommendation
  -------------------------- --------------------
  Young wizard story         Harry Potter
  Detective murder mystery   Seven
  Space exploration          Interstellar
  Epic love story            Titanic

Performance: - Average response time: \< 2 seconds - Dataset size: 1000
movies - Feature dimensions: 3000 TF‑IDF features

------------------------------------------------------------------------

## 🚀 Future Enhancements

-   Expand dataset to 10,000+ movies
-   Genre filtering
-   Year range selector
-   Rating filtering
-   Transformer models (BERT)
-   User accounts
-   REST API

------------------------------------------------------------------------

## 👥 Contributors

Your Name -- Subhashini G

------------------------------------------------------------------------

## 📄 License

MIT License

------------------------------------------------------------------------

## 🙏 Acknowledgments

-   Kaggle for dataset
-   Streamlit
-   scikit-learn
-   NLTK

------------------------------------------------------------------------