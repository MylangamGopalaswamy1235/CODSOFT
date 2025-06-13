import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
movies = pd.DataFrame({
    'title': [
        'The Shawshank Redemption', 'The Godfather', 'The Dark Knight',
        'Pulp Fiction', 'Forrest Gump', 'Inception',
        'The Matrix', 'Fight Club', 'The Silence of the Lambs', 'Titanic'
    ],
    'genres': [
        'Drama', 'Crime Drama', 'Action Crime Drama',
        'Crime Drama', 'Romance Drama', 'Action Adventure Sci-Fi',
        'Action Sci-Fi', 'Drama', 'Crime Drama Thriller', 'Romance Drama'
    ]
})
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies['genres'])
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
def recommend_movies(title, cosine_sim=cosine_sim):
    # Get index of the given movie
    idx = movies[movies['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]
print("Recommended Movies:")
print(recommend_movies('The Godfather'))
