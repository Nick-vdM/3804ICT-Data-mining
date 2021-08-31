import csv
import pandas as pd
import numpy as np

user_rating_filename = "../data/ratings.csv"
links_filename = "../data/links.csv"


def read_in_user_rating_csv():
    df = pd.read_csv(user_rating_filename)
    df = df.drop("timestamp", axis=1)
    return df


def read_in_links_csv():
    df = pd.read_csv(links_filename)
    df = df.drop("tmdbId", axis=1)
    return df


def map_movie_id_to_imdb_id(movie_id, links_dict=None):
    if links_df is None:
        return movie_id
    else:
        unpadded_id = str(links_dict[movie_id])
        padded_id = "tt" + '0'*(7-len(unpadded_id)) + unpadded_id
        return padded_id


if __name__ == "__main__":
    user_rating_df = read_in_user_rating_csv()
    print(user_rating_df.head(10))

    links_df = read_in_links_csv()
    print(links_df.head(10))

    imdb_ids = user_rating_df["movieId"].apply(map_movie_id_to_imdb_id, links_dict=links_df.set_index('movieId').to_dict()["imdbId"])
    user_rating_df = user_rating_df.assign(imdbId=imdb_ids)
    user_rating_df = user_rating_df.drop("movieId", axis=1)
    print(user_rating_df.head(10))


