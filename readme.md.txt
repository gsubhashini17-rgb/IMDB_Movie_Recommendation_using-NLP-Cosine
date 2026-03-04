# 🎬 IMDb Movie Recommendation System

## 📋 Overview
An intelligent movie recommendation system that suggests movies based on storyline similarity using Natural Language Processing.

## ✨ Features
- **Recommend by Storyline**: Input any movie plot and get similar movies
- **Recommend by Title**: Find movies similar to your favorite film
- **1000+ Movies**: Database of top IMDb movies
- **Real-time Similarity Scores**: See how similar each recommendation is
- **Interactive Interface**: Easy-to-use web app

## 🛠️ Tech Stack
- **Python** - Core programming language
- **Streamlit** - Web application framework
- **scikit-learn** - TF-IDF vectorization & cosine similarity
- **NLTK** - Text preprocessing (tokenization, lemmatization)
- **Pandas/NumPy** - Data manipulation
- **Plotly** - Interactive visualizations

## 🎯 How It Works
1. **Text Preprocessing**: Removes stopwords, punctuation, and lemmatizes text
2. **Feature Extraction**: Converts text to numerical vectors using TF-IDF
3. **Similarity Calculation**: Computes cosine similarity between vectors
4. **Recommendation**: Returns top 5 most similar movies

## 📊 Dataset
1000 top-rated movies from IMDb with:
- Movie titles
- Plot descriptions
- Release years
- IMDb ratings

## 🚀 Live Demo
[Click here to try the app](https://imdbmovierecommendationusing-nlp-cosine-pwtksz7t5pzkhcdx3yqzly.streamlit.app/)

## 📦 Installation (Local)
```bash
pip install -r requirements.txt

streamlit run app.py

