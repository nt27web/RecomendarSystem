import json
import requests

from scipy import sparse
from scipy.special._generate_pyx import generate_fused_type
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import sklearn.metrics.pairwise as pw
from IPython.display import display
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import numpy as np
import scipy
import pandas as pd
import math
import random
import sklearn
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse.linalg import svds
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

articles_df = pd.read_csv('shared_articles.csv')
articles_df = articles_df[articles_df['eventType'] == 'CONTENT SHARED']
articles_df.head(5)

interactions_df = pd.read_csv('users_interactions.csv')
print(interactions_df.shape)
print(interactions_df.head())

event_type_strength = {
    'VIEW': 1.0,
    'LIKE': 2.0,
    'BOOKMARK': 2.5,
    'FOLLOW': 3.0,
    'COMMENT CREATED': 4.0,
}

interactions_df['eventStrength'] = interactions_df['eventType'].apply(lambda x: event_type_strength[x])

users_interactions_count_df = interactions_df.groupby(['personId', 'contentId']).size().groupby('personId').size()
print('# users: %d' % len(users_interactions_count_df))
users_with_enough_interactions_df = users_interactions_count_df[users_interactions_count_df >= 5].reset_index()[
    ['personId']]
print('# users with at least 5 interactions: %d' % len(users_with_enough_interactions_df))

print('# of interactions: %d' % len(interactions_df))
interactions_from_selected_users_df = interactions_df.merge(users_with_enough_interactions_df,
                                                            how='right',
                                                            left_on='personId',
                                                            right_on='personId')
print('# of interactions from users with at least 5 interactions: %d' % len(interactions_from_selected_users_df))


def smooth_user_preference(x):
    return math.log(1 + x, 2)


interactions_full_df = interactions_from_selected_users_df \
        .groupby(['personId', 'contentId'])['eventStrength'].sum() \
        .apply(smooth_user_preference).reset_index()
print('# of unique user/item interactions: %d' % len(interactions_full_df))
interactions_full_df.head(10)

interactions_train_df, interactions_test_df = train_test_split(interactions_full_df,
                                                                   stratify=interactions_full_df['personId'],
                                                                   test_size=0.20,
                                                                   random_state=42)

print('# interactions on Train set: %d' % len(interactions_train_df))
print('# interactions on Test set: %d' % len(interactions_test_df))

# Ignoring stopwords (words with no semantics) from English and Portuguese
# (as we have a corpus with mixed languages)


stopwords_list = stopwords.words('english')  # + stopwords.words('portuguese')

# Trains a model whose vectors size is 5000, composed by the main unigrams and bigrams found in the corpus,
# ignoring stopwords
vectorizer = TfidfVectorizer(analyzer='word',
                             ngram_range=(1, 2),
                             min_df=0.003,
                             max_df=0.5,
                             max_features=5000,
                             stop_words=stopwords_list)

item_ids = articles_df['contentId'].tolist()
tfidf_matrix = vectorizer.fit_transform(articles_df['title'] + "" + articles_df['text'])
tfidf_feature_names = vectorizer.get_feature_names()
display(tfidf_matrix)


def get_item_profile(item_id):
    idx = item_ids.index(item_id)
    item_profile = tfidf_matrix[idx:idx + 1]
    return item_profile


def get_item_profiles(ids):
    item_profiles_list = [get_item_profile(x) for x in ids]
    item_profiles = scipy.sparse.vstack(item_profiles_list)
    return item_profiles


def build_users_profile(person_id, interactions_indexed_df):
    interactions_person_df = interactions_indexed_df.loc[person_id]
    user_item_profiles = get_item_profiles(interactions_person_df['contentId'])

    user_item_strengths = np.array(interactions_person_df['eventStrength']).reshape(-1, 1)
    # Weighted average of item profiles by the interactions strength
    user_item_strengths_weighted_avg = np.sum(user_item_profiles.multiply(user_item_strengths), axis=0) / np.sum(
        user_item_strengths)
    user_profile_norm = sklearn.preprocessing.normalize(user_item_strengths_weighted_avg)
    return user_profile_norm


def build_users_profiles():
    interactions_indexed_df = interactions_train_df[interactions_train_df['contentId'] \
        .isin(articles_df['contentId'])].set_index('personId')
    user_profiles = {}
    for person_id in interactions_indexed_df.index.unique():
        user_profiles[person_id] = build_users_profile(person_id, interactions_indexed_df)
    return user_profiles


def get_items_interacted(person_id):
    # Get the user's data and merge in the movie information.
    interacted_items = interactions_df.loc[person_id]['contentId']
    return set(interacted_items if type(interacted_items) == pd.Series else [interacted_items])


user_profiles = build_users_profiles()
print(len(user_profiles))


def _get_similar_items_to_user_profile(self, person_id, topn=10):
    # Computes the cosine similarity between the user profile and all item profiles
    cosine_similarities = cosine_similarity(user_profiles[person_id], tfidf_matrix)


def recommend_items(self, user_id, topn=10):
    similar_items = self._get_similar_items_to_user_profile(user_id)
    # Ignores items the user has already interacted
    similar_items_filtered = list(filter(lambda x: x[0] not in similar_items))

    recommendations_df = pd.DataFrame(similar_items_filtered, columns=['contentId', 'recStrength']) \
        .head(topn)

    recommendations_df = recommendations_df.merge(self.items_df
                                                  , how='left'
                                                  , left_on='contentId'
                                                  , right_on='contentId')[
            ['recStrength', 'contentId', 'title', 'url', 'lang']
    ]

    return recommendations_df


def _recommend():

    # myprofile = user_profiles[-1479311724257856983]
    # print(myprofile.shape)

    # pd.DataFrame(sorted(zip(tfidf_feature_names,
    #                         user_profiles[-1479311724257856983].flatten().tolist()), key=lambda x: -x[1])[:20],
    #              columns=['token', 'relevance'])

    recommend_items(-1479311724257856983)

    return 0

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

    articles_df = data_art[data_art['eventType'] == 'CONTENT SHARED']
    articles_df = data_art[data_art['lang'] == 'en']

    # articles_df = pd.DataFrame(articles_df, columns=['contentId', 'authorPersonId', 'content', 'lang'
    #     , 'title',	'text'
    # ])

    articles_df = pd.DataFrame(articles_df, columns=['contentId', 'authorPersonId', 'content', 'title', 'text'])

    # display(articles_df.head(10))

    # Define a TF-IDF Vectorizer Object. Remove all english stop words such as 'the', 'a'
    tfidf = TfidfVectorizer(stop_words='english')

    # Replace NaN with an empty string
    articles_df['text'] = articles_df['text'].fillna('')

    # Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(articles_df['text'])

    # Output the shape of tfidf_matrix
    display(tfidf_matrix.shape)

    # Array mapping from feature integer indices to feature name.
    # display(tfidf.get_feature_names()[5000:5010])

    # Compute the cosine similarity matrix
    # cosine_sim = pw.linear_kernel(tfidf_matrix, tfidf_matrix)
    # display(cosine_sim.shape)
    # display(cosine_sim[0])

    # Compute the cosine similarity matrix
    cosine_simi = cosine_similarity(tfidf_matrix, tfidf_matrix, True)
    # display(cosine_simi.shape)
    # display(cosine_simi[0])

    # Construct a reverse map of indices and movie titles
    indices = pd.Series(articles_df.index, index=articles_df['title']).drop_duplicates()
    # display(indices[:10])

    display(get_recommendations('Banks Need To Collaborate With Bitcoin and Fintech Developers', indices, cosine_simi
                                , articles_df))

    display(get_recommendations('Google Data Center 360° Tour', indices, cosine_simi,
                                articles_df))

    display(get_recommendations('The Rise And Growth of Ethereum Gets Mainstream Coverage', indices, cosine_simi,
                                articles_df))

    return 0


# Function that takes in movie title as input and outputs most similar movies
def get_recommendations(title, indices, cosine_sim, data):
    # Get the index of the article that matches the title
    idx = indices[title]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 most similar movies
    return data['title'].iloc[movie_indices]


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+Shift+B to toggle the breakpoint.


if __name__ == '__main__':
    # print_hi('PyCharm')
    _recommend()
