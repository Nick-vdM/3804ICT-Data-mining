import numpy as np
import pandas as pd
from proposal.useful_tools import pickle_manager


def get_truncated_data():
    movie_df: pd.DataFrame = pd.DataFrame(pickle_manager.load_pickle("../pickles/organised_movies.pickle.lz4"))

    # Shuffle it so that we're being fair
    movie_df = movie_df.sample(2000, random_state=42)
    print(movie_df.head())

    movies_that_exist = set()
    for index, row in movie_df.iterrows():
        movies_that_exist.add(row['imdbId'])

    rating_df: pd.DataFrame = pd.DataFrame(pickle_manager.load_pickle("../pickles/organised_ratings.pickle.lz4"))
    to_delete = []
    print("Before", len(rating_df))
    for index, row in rating_df.iterrows():
        if row['imdbId'] not in movies_that_exist:
            to_delete.append(index)

    print(to_delete)
    rating_df = rating_df.drop(to_delete)
    print("After", len(rating_df))

    return (movie_df, rating_df)


def embed_vectors(df, column_to_embed):
    pass


def convert_series_to_df(series):
    return pd.DataFrame.from_dict(dict(zip(series.index, series.values))).T


def append_columns(df1: pd.DataFrame, df2: pd.DataFrame,
                   title1='_a', title2='_g'):
    # Rename all of the columns in the first dataframe accordingly
    lookup = {}
    for column in list(df1.columns):
        lookup[column] = column + title1
    df1.rename(columns=lookup, inplace=True)
    # Merge them together with the second's name
    columns = list(df2.columns)
    for column_name in columns:
        df1[column_name + title2] = df2[column_name]
    return df1


def cosine_matrix_similarity(df):
    """
    Takes in an exploded matrix and calculates the matrix similarity
    :param df:
    :return:
    """
    matrix = np.zeros(shape=())
    pass


if __name__ == "__main__":
    movie_df, rating_df = get_truncated_data()
    movie_df['genre_features'] = embed_vectors(movie_df, 'genre')
    movie_df['actors_features'] = embed_vectors(movie_df, 'actors')
    features = append_columns(
        convert_series_to_df(movie_df['genre_features']),
        convert_series_to_df(movie_df['actors_df'])
    )
    cosine_matrix_similarity(features)
    pickle_manager.save_lzma_pickle(
        features, 'pickles/similarity_matrix.pickle.lz4'
    )
