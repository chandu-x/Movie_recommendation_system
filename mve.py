import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import ast  # Used to convert string representations of lists into actual lists

# Load the datasets
movies = pd.read_csv("C:/Users/chand/Downloads/ml project/movie_recommendation/tmdb_5000_movies.csv")
credits = pd.read_csv("C:/Users/chand/Downloads/ml project/movie_recommendation/tmdb_5000_credits.csv")

#  Merge movies and credits data on 'title'
movies = movies.merge(credits, on='title')

#  Select important columns
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Remove any missing values
movies.dropna(inplace=True)

#  Convert string-like lists into actual lists
def convert(text):
    try:
        data = ast.literal_eval(text)
        return [item['name'] for item in data]
    except:
        return []

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: convert(x)[:3])  # Get top 3 actors
movies['crew'] = movies['crew'].apply(lambda x: [i['name'] for i in ast.literal_eval(x) if i['job'] == 'Director'])

# Convert overview (summary) into a list of words
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Merge all selected features into a single 'tags' column
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
movies['tags'] = movies['tags'].apply(lambda x: " ".join(x).lower())  # Convert to lowercase

# Convert text into numerical format using Bag of Words
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(movies['tags']).toarray()

#  Compute similarity between movies
similarity = cosine_similarity(vectors)

# Define a function to recommend similar movies
def recommend(movie_name):
    if movie_name not in movies['title'].values:
        return "Movie not found. Please check the spelling."

    movie_index = movies[movies['title'] == movie_name].index[0]
    distances = similarity[movie_index]
    
    # Get top 10 similar movies
    similar_movies = sorted(enumerate(distances), reverse=True, key=lambda x: x[1])[1:11]

    recommendations = []
    for i in similar_movies:
        recommendations.append(movies.iloc[i[0]].title)
    
    return recommendations

# Example usage
enter_movie_name=input()
print(recommend(enter_movie_name))
