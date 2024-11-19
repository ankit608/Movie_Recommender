import kagglehub
import numpy as np
import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

ps = PorterStemmer()


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# Download latest version
# C:\Users\ANKIT SAHU\.cache\kagglehub\datasets\tmdb\tmdb-movie-metadata\versions\2
path = kagglehub.dataset_download("tmdb/tmdb-movie-metadata")

movies = pd.read_csv("tmdb_5000_movies.csv")
credit = pd.read_csv("tmdb_5000_credits.csv")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 20000)
print(movies.head())
print(
    "############################################################################################################################## #################")


movies = movies.merge(credit, on="title")



# genres,#id,#keywords,#overview#

print("Path to dataset files:", path)


movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

movies.dropna(inplace=True)



def convert(obj):
    l = []
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l


def convert2(obj):
    l = []
    count = 0
    for i in ast.literal_eval(obj):
        if count != 3:
            l.append(i['name'])
            count = count + 1
        else:
            break
    return l


def fetch_director(obj):
    l = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            l.append(i["name"])
            break
    return l


movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert2)
movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['overview'] = movies['overview'].apply(lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ", "") for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id', 'title', 'tags']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())


new_df['tags'] = new_df['tags'].apply(stem)
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()


similarity = cosine_similarity(vectors)
def recommended(movie):
    moive_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[moive_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


recommended('Batman Begins')
