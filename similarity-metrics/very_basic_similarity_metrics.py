import numpy as np
import pandas as pd
from proposal.useful_tools import pickle_manager
from sklearn.metrics.pairwise import cosine_similarity


def get_truncated_data():
    movie_df: pd.DataFrame = pd.DataFrame(pickle_manager.load_pickle("../pickles/organised_movies.pickle.lz4"))

    # Shuffle it so that we're being fair
    movie_df = movie_df.sample(2000, random_state=42)

    movies_that_exist = set()
    for index, row in movie_df.iterrows():
        movies_that_exist.add(row['imdbId'])

    rating_df: pd.DataFrame = pd.DataFrame(pickle_manager.load_pickle("../pickles/organised_ratings.pickle.lz4"))
    to_delete = []
    for index, row in rating_df.iterrows():
        if row['imdbId'] not in movies_that_exist:
            to_delete.append(index)

    rating_df = rating_df.drop(to_delete)

    return (movie_df, rating_df)


def embed_vectors(df, column_to_embed):
    lookup = {}
    for items in df[column_to_embed].values:
        for item in items:
            if item not in lookup:
                lookup[item] = len(lookup)

    series = []

    for items in df[column_to_embed].values:
        vector = [0] * len(lookup)
        for item in items:
            vector[lookup[item]] = 1
        series.append(vector)

    return pd.DataFrame(series)


def average_user(
        embedded_movies_df: pd.DataFrame,
        set_of_movies_to_average: int,
        movies_id_lookup: dict
):
    """
    Averages a given user
    :param embedded_movies_df:
    :param set_of_movies_to_average:
    :param movies_id_lookup:
    :return:
    """
    # TODO
    # Get all of the movies the user's seen
    pass


def average_all_user_movies(
        ratings_df: pd.DataFrame,
        embedded_movies_df: pd.DataFrame,
        movies_id_lookup: dict
):
    """
    Generates a dataframe of features that describes a user's average movie
    :param ratings_df:
    :param embedded_movies_df:
    :param movies_id_lookup:
    :return:
    """
    # defragmenting it is going to make the next section a little faster
    ratings_df = ratings_df.groupby(['userId'])
    users = ratings_df['userId'].unique()

    embedded_users = pd.DataFrame(
        columns=embedded_movies_df.columns
    )

    user_lookup = {}

    for user in users:
        watched_movies = ratings_df.query('userId ==', user)['imdbId'].unique()

        embedded_users.append(
            average_user(embedded_movies_df,
                         set_of_movies_to_average=watched_movies,
                         movies_id_lookup=movies_id_lookup
                         )
        )

    return embedded_users, user_lookup


def append_columns(df1: pd.DataFrame, df2: pd.DataFrame,
                   title1='_a', title2='_g'):
    # Rename all of the columns in the first dataframe accordingly
    lookup = {}
    for column in list(df1.columns):
        lookup[column] = str(column) + title1
    df1.rename(columns=lookup, inplace=True)
    # Merge them together with the second's name
    columns = list(df2.columns)
    for column_name in columns:
        df1[str(column_name) + title2] = df2[column_name]
    return df1


if __name__ == "__main__":
    movie_df, rating_df = get_truncated_data()
    genre_features: pd.DataFrame = embed_vectors(movie_df, 'genre')
    actor_features: pd.DataFrame = embed_vectors(movie_df, 'actors')
    movies_features = append_columns(
        genre_features, actor_features
    )
    movies_index_lookup = {}
    # Generate a lookup of index : movieId
    for index, row in movie_df.iterrows():
        movies_index_lookup[row['imdbId']] = index

    ratings_features = average_all_user_movies(
        rating_df,
        movies_features,
        movies_index_lookup
    )

    print(movies_features.head())
    movies_similarity_matrix = cosine_similarity(movies_features)

    pickle_manager.save_lzma_pickle(
        movies_similarity_matrix, '../pickles/similarity_matrix.pickle.lz4'
    )

    pickle_manager.save_lzma_pickle(
        movie_df, '../pickles/simil_movies.pickle.lz4'
    )
    pickle_manager.save_lzma_pickle(
        rating_df, '../pickles/simil_ratings.pickle.lz4'
    )
