import csv
import pandas as pd
import numpy as np
import pickle
import bz2

user_rating_filename = "../data/clean_ratings.csv"
links_filename = "../data/clean_links.csv"
movies_filename = "../data/clean_IMDb movies.csv"

input_files = ["../data/ratings.csv", "../data/links.csv", "../data/IMDb movies.csv"]

user_rating_clean_filename = "../data/clean_ratings.csv"

user_rating_output_filename = "../data/organised_ratings.csv"
movies_output_filename = "../data/organised_movies.csv"

user_rating_pickle_filename = "../pickles/organised_ratings.pickle"
movies_pickle_filename = "../pickles/organised_movies.pickle"


def remove_trailing_spaces():
    for filename in input_files:
        with open(filename, 'r', encoding='utf8', newline='') as csv_input:
            reader = csv.reader(csv_input)
            with open(filename[:8] + "clean_" + filename[8:], 'w', encoding='utf8', newline='') as csv_output:
                writer = csv.writer(csv_output)
                for row in reader:
                    row = [col.strip() for col in row]
                    writer.writerow(row)


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


def read_in_movies_csv():
    movies_dtypes = {
        "imdb_title_id": str,
        "title": str,
        "original_title": str,
        "year": str,  # Some fields say "TV Movie YYYY"
        "date_published": str,
        "genre": str,
        "duration": int,
        "country": str,
        "language": str,
        "director": str,
        "writer": str,
        "production_company": str,
        "actors": str,
        "avg_vote": float,
        "votes": int,
        "budget": str,
        "usa_gross_income": str,
        "worlwide_gross_income": str,
        "metascore": float,
        "reviews_from_users": float,
        "reviews_from_critics": float
    }
    df = pd.read_csv(movies_filename, dtype=movies_dtypes)
    return df


def map_str_to_set(input_val):
    input_str = str(input_val)
    if input_str is None:
        return set()
    else:
        final_set = set()
        for entry in input_str.split(sep=", "):
            final_set.add(entry)
        return final_set


def fix_strings_in_int_fields(string_input):
    if string_input[:9] == "TV Movie ":
        return int(string_input[9:])
    else:
        return int(string_input)


def convert_float_to_int_fields(float_val):
    if np.isnan(float_val):
        return -1
    else:
        return int(float_val)


if __name__ == "__main__":
    remove_trailing_spaces()

    user_rating_df = read_in_user_rating_csv()
    print(user_rating_df.head(10))

    links_df = read_in_links_csv()
    print(links_df.head(10))

    imdb_ids = user_rating_df["movieId"].apply(map_movie_id_to_imdb_id, links_dict=links_df.set_index('movieId').to_dict()["imdbId"])
    user_rating_df = user_rating_df.assign(imdbId=imdb_ids)
    user_rating_df = user_rating_df.drop("movieId", axis=1)
    print(user_rating_df.head(10))
    user_rating_df.to_csv(user_rating_output_filename)

    pickle_file = bz2.BZ2File(user_rating_pickle_filename, "w")
    pickle.dump(user_rating_df, pickle_file)
    pickle_file.close()

    movies_df = read_in_movies_csv()

    movie_genre_sets = movies_df["genre"].apply(map_str_to_set)
    movies_df = movies_df.drop("genre", axis=1)
    movies_df = movies_df.assign(genre=movie_genre_sets)

    movie_actor_sets = movies_df["actors"].apply(map_str_to_set)
    movies_df = movies_df.drop("actors", axis=1)
    movies_df = movies_df.assign(actors=movie_actor_sets)

    movie_years = movies_df["year"].apply(fix_strings_in_int_fields)
    movies_df = movies_df.drop("year", axis=1)
    movies_df = movies_df.assign(year=movie_years)

    movies_metascore = movies_df["metascore"].apply(convert_float_to_int_fields)
    movies_df = movies_df.drop("metascore", axis=1)
    movies_df = movies_df.assign(metascore=movies_metascore)

    movies_reviews_from_users = movies_df["reviews_from_users"].apply(convert_float_to_int_fields)
    movies_df = movies_df.drop("reviews_from_users", axis=1)
    movies_df = movies_df.assign(reviews_from_users=movies_reviews_from_users)

    movies_reviews_from_critics = movies_df["reviews_from_critics"].apply(convert_float_to_int_fields)
    movies_df = movies_df.drop("reviews_from_critics", axis=1)
    movies_df = movies_df.assign(reviews_from_critics=movies_reviews_from_critics)

    print(movies_df.head(10))
    movies_df.to_csv(movies_output_filename)

    pickle_file = bz2.BZ2File(movies_pickle_filename, "w")
    pickle.dump(movies_df, pickle_file)
    pickle_file.close()
