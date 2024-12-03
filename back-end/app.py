from flask import Flask, jsonify, request
import pandas as pd
import torch
import pickle
import heapq
import cloudpickle

root = "./back-end/"

class Model(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=20):
        super().__init__()
        # create user and item embeddings
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.movie_factors = torch.nn.Embedding(n_items, n_factors)
        # fills weights with values from a uniform distribution [0, 0.5]
        self.user_factors.weight.data.uniform_(0, 0.05)
        self.movie_factors.weight.data.uniform_(0, 0.05)
    
    def forward(self, data):
        # matrix multiplication between user and item factors, and then concatenates them to one column
        return (self.user_factors(data[:,0])*self.movie_factors(data[:,1])).sum(1)
    
    def predict(self, user, item):
        return (self.user_factors(user)*self.movie_factors(item)).sum(1)

def get_top_n_recommendations(i, n):
    movies_df = pd.read_csv(root + "Data/ml-latest-small/movies.csv")
    with open(root + 'recSys.pkl', 'rb') as f:
        model = pickle.load(f)
    # Load in data from csv files
    actual_ratings = pd.read_csv(root + "Data/ml-latest-small/ratings.csv")
    actual_ratings = actual_ratings.pivot(index='userId', columns='movieId', values='rating').fillna(-1)
    predicted_ratings = torch.matmul(model.user_factors.weight.data, model.movie_factors.weight.data.T)

    # i < actual_ratings.shape[0]

    recommendations = []
    rated = []
    for j in range(actual_ratings.shape[1]):
        if actual_ratings.iloc[i][actual_ratings.columns[j]] > 0:
            rated.append((actual_ratings.columns[j], actual_ratings.iloc[i][actual_ratings.columns[j]], float(predicted_ratings[i][j])))
        else:
            heapq.heappush(recommendations, (-float(predicted_ratings[i][j]), actual_ratings.columns[j]))
    # difference between predicted and rated: sum(list(map(lambda x: (x[1] - x[2])**2, rated)))
    recs = []
    movie_names = movies_df.set_index('movieId')['title'].to_dict()
    for x in range(n):
        recs.append(movie_names[recommendations[x][1]])
    return recs

def translate_ratings(movie_ratings):
    import re
    movies_df = pd.read_csv(root + "Data/ml-latest-small/movies.csv")
    movies_df = movies_df[['title', 'movieId']]
    movies_df['year'] = pd.to_numeric(movies_df['title'].str.strip("\"").str[-5:-1])
    movies_df['title'] = movies_df['title'].str.strip("\"").str[:-6].str.strip()
    movies_df['title'] = movies_df['title'].apply(lambda t : "The " + t[:-5] if t[-5:] == ", The" else t)
    movies_df['title'] = movies_df['title'].str.replace(r'[^a-zA-Z0-9 -]+', '', regex=True)
    movies_df['title'] = movies_df['title'].str.split(" ")

    new_ratings = []
    for rating in movie_ratings:
        similarity = movies_df.copy(deep=True)
        similarity = similarity[similarity['year'] == rating[1]]
        similarity['similarity'] = similarity['title'].apply(lambda x: len(set(x).intersection(re.sub(r'[^a-zA-Z -]+', '', rating[0]).split(" "))))
        # print(similarity.loc[similarity['similarity'].idxmax()])
        new_ratings.append([similarity['similarity'].idxmax(), rating[2]])
    return new_ratings

def get_similar_users(movie_ratings):
    user_vector = pd.array([-1.0] * 9724)
    ratings = translate_ratings(movie_ratings)
    for i in ratings:
        user_vector[[i[0]]] = i[1]
    with open(root + 'userRec.pkl', 'rb') as f:
        model = cloudpickle.load(f) 
    return model.kneighbors([user_vector], 5, return_distance=False)[0]

def get_movie_recommendations(users):
    recs = {}
    for userId in users:
        for user_rec in get_top_n_recommendations(userId, 20):
            recs[user_rec] = recs.get(user_rec, 0) + 1
    return sorted(recs.keys(), key=lambda x: -1 if int(x[-5:-1]) >= 2000 else recs[x], reverse=True)

import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

cred = credentials.Certificate("./back-end/serviceAccountKey.json")
firebase_admin.initialize_app(cred)

db = firestore.client()

app = Flask(__name__)

# Sample data to send to the frontend
sample_data = {
    "movies": [
        {"title": "Inception", "cover": "avo.jpg"},
        {"title": "The Matrix", "cover": "avo.jpg"},
        {"title": "Interstellar", "cover": "avo.jpg"}
    ]
}

@app.route("/")
def default():
    return {}

@app.route("/get/<int:i>/<int:n>")
def get_recommendations(i, n):
    return get_top_n_recommendations(i, n)

@app.route("/movies")
def movies():
    return get_movie_recommendations(get_similar_users([["Whiplash", 2014, 4.3], ["A Quiet Place", 2018, 5], ["Inception", 2010, 4.8], ["Ready Player One", 2018, 4.5], ["The Devil Wears Prade", 2006, 5]]))

@app.route("/user")
def similar_users():
    return get_similar_users([["c2", 1999, 5], ["Whiplash", 2014, 4.3], ["The Artist", 2011, 3.4], ["Delivery Man", 2013, 4.3], ["The Devil Wears Prade", 2006, 5]])

@app.route("/user/<string:x>")
def get(x: str):
    doc = db.collection("users").document(x).get()
    if doc.exists:
        ratings = []
        for rating in doc.to_dict()["reviews"]:
            ratings.append([rating["name"], int(rating["year"]), float(rating["rating"])])
        print(ratings)
    return get_movie_recommendations(get_similar_users(ratings))

@app.route("/test")
def test():
    return {"sample result fetched": "test"}

@app.route('/api/movies', methods=['GET'])
def get_movies():
    return jsonify(sample_data)

# New POST endpoint to receive movie name and keywords
# @app.route('/api/add_movie', methods=['POST'])
# def add_movie():
#     data = request.get_json()
#     movie_title = data.get('title')
#     movie_keywords = data.get('keywords')

#     # You can add logic here to process or store the received data
#     print(f"Received movie: {movie_title} with keywords: {movie_keywords}")

#     # Send a response back to the frontend
#     return jsonify({"message": "Movie received successfully!"})

if __name__ == '__main__':
    app.run(debug=True)

# if __name__ == "__main__":
#     app.run(host='127.0.0.1', port=8080, debug=True)