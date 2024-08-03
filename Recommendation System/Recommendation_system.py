import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Sample dataset
data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3],
    'movie_id': [101, 102, 103, 101, 104, 102, 103, 104],
    'rating': [4, 5, 2, 5, 4, 2, 5, 3]
}

# Create DataFrame
df = pd.DataFrame(data)

# Pivot table to create a user-item matrix
user_movie_matrix = df.pivot_table(index='user_id', columns='movie_id', values='rating')

# Fill NaN values with 0 (indicating no rating)
user_movie_matrix.fillna(0, inplace=True)

print("User-Movie Matrix:")
print(user_movie_matrix)

# Calculate cosine similarity between users
user_similarity = cosine_similarity(user_movie_matrix)

# Convert to DataFrame for better readability
user_similarity_df = pd.DataFrame(user_similarity, index=user_movie_matrix.index, columns=user_movie_matrix.index)

print("\nUser Similarity Matrix:")
print(user_similarity_df)

def recommend_movies(user_id, user_movie_matrix, user_similarity_df, num_recommendations=3):
    # Get the similarity scores for the user
    user_sim_scores = user_similarity_df.loc[user_id]
    
    # Get the movies the user has already rated
    user_rated_movies = user_movie_matrix.loc[user_id]
    
    # Find similar users
    similar_users = user_sim_scores.sort_values(ascending=False).index[1:]
    
    # Collect movies rated by similar users
    recommended_movies = {}
    
    for similar_user in similar_users:
        similar_user_ratings = user_movie_matrix.loc[similar_user]
        
        # Recommend movies the current user hasn't rated yet
        for movie_id, rating in similar_user_ratings.iteritems():
            if user_rated_movies[movie_id] == 0 and rating > 0:  # Only consider movies not yet rated by the user
                if movie_id not in recommended_movies:
                    recommended_movies[movie_id] = rating
                else:
                    recommended_movies[movie_id] += rating
    
    # Sort recommended movies by aggregated rating
    recommended_movies = sorted(recommended_movies.items(), key=lambda x: x[1], reverse=True)
    
    # Return top N recommendations
    return [movie_id for movie_id, rating in recommended_movies[:num_recommendations]]

# Example: Recommend movies for user 1
recommended_movies_for_user_1 = recommend_movies(1, user_movie_matrix, user_similarity_df)
print(f"\nRecommended movies for user 1: {recommended_movies_for_user_1}")

# Example: Recommend movies for user 2
recommended_movies_for_user_2 = recommend_movies(2, user_movie_matrix, user_similarity_df)
print(f"Recommended movies for user 2: {recommended_movies_for_user_2}")

# Example: Recommend movies for user 3
recommended_movies_for_user_3 = recommend_movies(3, user_movie_matrix, user_similarity_df)
print(f"Recommended movies for user 3: {recommended_movies_for_user_3}")
