import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask
from flask import render_template
from flask import request
import requests
import json

app = Flask(__name__)
@app.route("/", methods = ["GET", "POST"])

def main():

    if request.method == "GET":
        return render_template("index.html")

    elif request.method == "POST":

        recommended_movies = []
        movie_posters = []
        imdb_url = []

        # retrive input from search box
        value = request.form.get("search", type = str)
        movie_name = str(value)

        movies_data = pd.read_csv('movies.csv')
        selected_features = ['genres','keywords','tagline','cast','director']

        # replacing the null valuess with null string
        for feature in selected_features:
            movies_data[feature] = movies_data[feature].fillna('')

        # combining all the 5 selected features
        combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']

        # converting the text data to feature vectors
        vectorizer = TfidfVectorizer()
        feature_vectors = vectorizer.fit_transform(combined_features)

        # getting the similarity scores using cosine similarity
        similarity = cosine_similarity(feature_vectors)

        list_of_all_titles = movies_data['title'].tolist()
        
        find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
        
        close_match = find_close_match[0]
        
        index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

        similarity_score = list(enumerate(similarity[index_of_the_movie]))

        sorted_similar_movies = sorted(similarity_score, key = lambda x:x[1], reverse = True)

        i = 1

        for movie in sorted_similar_movies:
            index = movie[0]
            title_from_index = movies_data[movies_data.index==index]['title'].values[0]
            if (i<=10):
                recommended_movies.append(title_from_index)
                i+=1
        
        for movie in recommended_movies:
            m = movie
            m = m.replace(" ", "+")
            url = "http://www.omdbapi.com/?t="+m+"&apikey=6d31dee3"
            json_file = json.loads(requests.get(url).text)
            poster = json_file["Poster"]
            u = json_file["imdbID"]
            imdb_u = "https://www.imdb.com/title/"+u
            movie_posters.append(poster)
            imdb_url.append(imdb_u)
        
        return render_template("movies.html", movies = recommended_movies, posters = movie_posters, imdb = imdb_url)
        

if __name__ == "__main__":
    app.debug = True
    app.run()

