import pandas as pd
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import json
from ast import literal_eval

import requests

from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

import numpy as np
import sklearn.metrics.pairwise as pw

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import scipy

import math
import random
import sklearn
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split

from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def _recommend():
    # articles_df = pd.read_csv('shared_articles.csv')
    #
    #
    # articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
    # articles_df.head(5)
    #
    # interactions_df = pd.read_csv('users_interactions.csv')
    # print(interactions_df.shape)
    # print(interactions_df.head())

    # display(data_art.head())
    # display(len(data_art))
    # articles_df.head()
    # display(len(articles_df))
    # display(data_art.columns)
    # display(data_art['lang'].unique())
    # display(data_art['lang'].isnull().sum())
    #
    # display(len(data_art['contentId'].unique()))
    # display(data_art['contentId'].isnull().sum())
    #
    # display(len(data_art['authorPersonId'].unique()))
    # display(data_art['authorPersonId'].isnull().sum())
    #
    # display(len(data_art['authorRegion'].unique()))
    # display(data_art['authorRegion'].isnull().sum())
    #
    # display(data_art['authorCountry'].unique())
    # display(data_art['authorCountry'].isnull().sum())
    #
    # display(data_art['contentType'].unique())
    # display(data_art['contentType'].isnull().sum())
    #
    # display(len(data_art['title'].unique()))
    # display(data_art['title'].isnull().sum())
    #
    # display(len(data_art['text'].unique()))
    # display(data_art['text'].isnull().sum())

    articles_df = pd.read_csv('shared_articles.csv')
    print(articles_df.shape)
    # print(articles_df.head(5))
    # print(articles_df['timestamp'].head(5))
    # print(articles_df['eventType'].value_counts())

    articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
    # print(articles_df.shape)
    # print(articles_df['contentId'].head(5))
    # print(articles_df['authorPersonId'].head(5))
    # print(len(articles_df['authorPersonId'].unique()))
    # print(articles_df['authorSessionId'].head(5))
    # print(articles_df['authorUserAgent'].tail(5))
    # print(len(articles_df['authorUserAgent'].unique()))
    # print(articles_df['authorRegion'].tail(5))
    # print(len(articles_df['authorRegion'].unique()))
    # print(articles_df['authorRegion'].isnull().sum(axis=0))
    # print(articles_df['authorRegion'].isna.sum(axis=0))

    # print(articles_df['authorCountry'].tail(5))
    # print(articles_df['authorCountry'].unique())
    # print(articles_df['authorCountry'].isnull().sum(axis=0))
    # print(articles_df['authorCountry'].isna.sum(axis=0))

    # print(articles_df['contentType'].tail(5))
    # print(articles_df['contentType'].unique())
    # print(articles_df['contentType'].isnull().sum(axis=0))
    # print(articles_df['contentType'].isna.sum(axis=0))

    # print(articles_df['url'].head(5))
    # print(articles_df['url'].isnull().sum(axis=0))
    # print(articles_df['url'].isna().sum(axis=0))

    # print(articles_df['title'].head(5))
    # print(articles_df['title'].isnull().sum(axis=0))
    # print(articles_df['title'].isna().sum(axis=0))

    # print(articles_df['text'].head(5))
    # print(articles_df['text'].isnull().sum(axis=0))
    # print(articles_df['text'].isna().sum(axis=0))

    # print(articles_df['lang'].unique())
    # print(articles_df['lang'].isnull().sum(axis=0))
    # print(articles_df['lang'].isna().sum(axis=0))

    # Replace NaN with an empty string
    # articles_df['text'] = articles_df['text'].fillna('')

    articles_df = articles_df[articles_df['lang'] == 'en']
    # print(articles_df.shape)

    # return 0

    # articles_df = pd.DataFrame(articles_df, columns=['contentId', 'authorPersonId', 'content', 'lang'
    #     , 'title',	'text'
    # ])

    articles_df = pd.DataFrame(articles_df, columns=['contentId', 'authorPersonId', 'content', 'title', 'text'])

    # display(len(articles_df))
    # display(articles_df.head(5))

    # interactions_df = pd.DataFrame(interactions_df, columns=['contentId', 'userCountry']).drop_duplicates(inplace=True)

    # articles_df['contentId'] = articles_df['contentId'].astype('int')
    # interactions_df['contentId'] = interactions_df['contentId'].astype('int')

    # Merge keywords and credits into your main metadata dataframe
    # metadata = articles_df.merge(interactions_df.set_index('contentId'), on='contentId', how="inner")

    # display(len(metadata))
    # display(metadata.head())
    # display(metadata.columns)

    # return 0
    # metadata.drop('timestamp', inplace=True, axis=1)
    # metadata.drop('sessionId', inplace=True, axis=1)
    # metadata.drop('userAgent', inplace=True, axis=1)
    # metadata.drop('userRegion', inplace=True, axis=1)
    # metadata.drop('personId', inplace=True, axis=1)
    # metadata.drop('content', inplace=True, axis=1)
    # metadata.drop('eventType', inplace=True, axis=1)
    # # metadata.drop('contentId', inplace=True, axis=1)
    # # display(metadata.columns)
    # metadata.drop('authorPersonId', inplace=True, axis=1)
    # metadata.dropna(subset=["userCountry"], inplace=True)


    # display(metadata.columns)
    # display(metadata.head(5))
    # return 0

    # features = ['text']
    # display(metadata.columns)

    # for feature in features:
    #     metadata[feature] = metadata[feature].apply(literal_eval)

    # metadata['soup'] = metadata.apply(create_soup, axis=1)
    articles_df['soup'] = articles_df.apply(create_soup, axis=1)
    # display(metadata.head(5))
    # display(metadata['soup'].head(5))
    # display(metadata[['soup']].head(2))

    # return 0

    # display(metadata.head(10))

    # Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')

    # Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(articles_df['text'])

    # Output the shape of tfidf_matrix
    # print(tfidf_matrix.shape)
    # print(tfidf.get_stop_words().pop())

    # Array mapping from feature integer indices to feature name.
    # print(tfidf.get_feature_names()[5000:5010])

    # count vectorize
    # count = CountVectorizer(stop_words='english')
    # count_matrix = count.fit_transform(metadata['soup'])

    # display(count_matrix.shape)
    # return 0
    # Compute the cosine similarity matrix

    # cosine_sim = cosine_similarity(count_matrix, count_matrix, True)

    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix, True)
    # display(cosine_sim.shape)
    # display(cosine_sim)

    # Construct a reverse map of indices and movie titles
    # Reset index of your main DataFrame and construct reverse mapping as before
    metadata = articles_df.reset_index()
    # indices = pd.Series(metadata.index, index=metadata['title'])
    indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()
    # display(indices[:10])

    # return 0

    print(get_recommendations('The Rise And Growth of Ethereum Gets Mainstream Coverage', indices, cosine_sim,
                                metadata))

    return 0
    print(get_recommendations('Google Data Center 360° Tour', indices, cosine_sim,
                                metadata))

    # return 0

    print(get_recommendations('Intel\'s internal IoT platform for real-time enterprise analytics', indices, cosine_sim
                                , metadata))


    return 0


def create_soup(x):
    soup = ' '.join(x['text'])
    return soup


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, indices, cosine_sim, data):
    # Get the index of the article that matches the title
    idx = indices[title]
    # print(idx)
    # return 0
    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # print(sim_scores)
    # return 0
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # print(sim_scores)
    # return 0
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]
    # print(sim_scores)
    # return 0
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    # print(movie_indices)
    # return 0
    # Return the top 10 most similar movies
    return data['title'].iloc[movie_indices]


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+Shift+B to toggle the breakpoint.


if __name__ == '__main__':
    # print_hi('PyCharm')
    _recommend()
