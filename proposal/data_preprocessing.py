import csv
import os

import pandas as pd
import numpy as np
import compress_pickle


class Preprocessor:
    """
    Handles merging the CSV files and cleaning it as much as possible
    Uses datasets from https://www.kaggle.com/stefanoleone992/imdb-extensive-dataset
    and https://grouplens.org/datasets/movielens/latest/
    """

    def __init__(self):
        """
        Define where all the files are sitting
        """
        self.user_rating_filename = "../data/clean_ratings.csv"
        self.links_filename = "../data/clean_links.csv"
        self.movies_filename = "../data/clean_IMDb movies.csv"

        self.input_files = ["../data/ratings.csv", "../data/links.csv", "../data/IMDb movies.csv"]

        self.user_rating_clean_filename = "../data/clean_ratings.csv"

        self.user_rating_output_filename = "../data/organised_ratings.csv"
        self.movies_output_filename = "../data/organised_movies.csv"

        self.user_rating_pickle_filename = "../pickles/organised_ratings.pickle.lzma"
        self.movies_pickle_filename = "../pickles/organised_movies.pickle.lzma"

    def assign_types_and_merge(self):
        """
        Does some preliminary, required cleaning and merges the classes together
            1. Clears trailing spaces in all the csv files
            2. Creates a movie rating pickle
                a. Merges it with links so we know which movies it is
        :return: None
        """
        print("Beginning Merge")
        print("Stage 1: Clear trailing spaces from csvs")
        self._remove_trailing_spaces()

        print("Stage 2: Reading in user ratings csv")
        user_rating_df = self._read_in_user_rating_csv()

        user_rating_df = self._merge_user_ratings(user_rating_df)

        self._save_dataframe(
            user_rating_df,
            self.user_rating_output_filename,
            self.user_rating_pickle_filename
        )

        print("Stage 3: Starting to process movies file")
        movies_df = self._read_in_movies_csv()
        movies_df = self._set_movie_data_types(movies_df)
        self._save_dataframe(movies_df,
                             self.movies_output_filename,
                             self.movies_pickle_filename
                             )
        print('Completed merge')

    def clean_noisy_data(self):
        """
        This assumes the pickle file has been created, and we can look at cleaning
        the noisy data. Saves files as a separate pickle as the merge function
        :return: None
        """
        print("Starting cleaning noisy data")
        pass
        print("Finished cleaning noisy data")

    def _merge_user_ratings(self, user_rating_df):
        links_df = self._read_in_links_csv()
        print(links_df.head(10))
        imdb_ids = user_rating_df["movieId"].apply(
            self._map_movie_id_to_imdb_id,
            links_dict=links_df.set_index('movieId').to_dict()["imdbId"]
        )
        user_rating_df = user_rating_df.assign(imdbId=imdb_ids)
        user_rating_df = user_rating_df.drop("movieId", axis=1)
        print(user_rating_df.head(10))
        return user_rating_df

    def _set_movie_data_types(self, movies_df):
        # Since these require a kwarg operator

        movie_genre_sets = movies_df["genre"].apply(self._map_str_to_set)
        movies_df = movies_df.drop("genre", axis=1)
        movies_df = movies_df.assign(genre=movie_genre_sets)

        movie_actor_sets = movies_df["actors"].apply(self._map_str_to_set)
        movies_df = movies_df.drop("actors", axis=1)

        movies_df = movies_df.assign(actors=movie_actor_sets)
        movie_years = movies_df["year"].apply(self._fix_strings_in_int_fields)
        movies_df = movies_df.drop("year", axis=1)

        movies_df = movies_df.assign(year=movie_years)
        movies_metascore = movies_df["metascore"].apply(self._convert_float_to_int_fields)
        movies_df = movies_df.drop("metascore", axis=1)

        movies_df = movies_df.assign(metascore=movies_metascore)
        movies_reviews_from_users = movies_df["reviews_from_users"].apply(self._convert_float_to_int_fields)
        movies_df = movies_df.drop("reviews_from_users", axis=1)

        movies_df = movies_df.assign(reviews_from_users=movies_reviews_from_users)
        movies_reviews_from_critics = movies_df["reviews_from_critics"].apply(self._convert_float_to_int_fields)
        movies_df = movies_df.drop("reviews_from_critics", axis=1)
        movies_df = movies_df.assign(reviews_from_critics=movies_reviews_from_critics)

        return movies_df

    def _remove_trailing_spaces(self):
        for filename in self.input_files:
            print("\t Doing", filename)
            with open(filename, 'r', encoding='utf8', newline='') as csv_input:
                reader = csv.reader(csv_input)
                with open(filename[:8] + "clean_" + filename[8:], 'w', encoding='utf8', newline='') as csv_output:
                    writer = csv.writer(csv_output)
                    for row in reader:
                        row = [col.strip() for col in row]
                        writer.writerow(row)

    def _read_in_user_rating_csv(self):
        df = pd.read_csv(self.user_rating_filename)
        df = df.drop("timestamp", axis=1)
        return df

    def _read_in_links_csv(self):
        df = pd.read_csv(self.links_filename)
        df = df.drop("tmdbId", axis=1)
        return df

    def _read_in_movies_csv(self):
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
        df = pd.read_csv(self.movies_filename, dtype=movies_dtypes)
        return df

    @staticmethod
    def _map_movie_id_to_imdb_id(movie_id, links_dict=None):
        if links_dict is None:
            return movie_id
        unpadded_id = str(links_dict[movie_id])
        padded_id = "tt" + '0' * (7 - len(unpadded_id)) + unpadded_id
        return padded_id

    @staticmethod
    def _map_str_to_set(input_val):
        input_str = str(input_val)

        if input_str is None:
            return set()

        final_set = set()
        for entry in input_str.split(sep=", "):
            final_set.add(entry)
        return final_set

    @staticmethod
    def _fix_strings_in_int_fields(string_input):
        if string_input[:9] == "TV Movie ":
            return int(string_input[9:])
        return int(string_input)

    @staticmethod
    def _convert_float_to_int_fields(float_val):
        if np.isnan(float_val):
            return -1
        return int(float_val)

    @staticmethod
    def _save_dataframe(df_to_save, csv_filename, pickle_filename):
        df_to_save.to_csv(csv_filename)
        # This is an excerpt from Nick's SEEKAnalysis assignment for compressed
        # pickles
        try:
            f = open(pickle_filename, 'wb')
        except FileNotFoundError:
            # The specified path probably doesn't exist
            # Use mkdir and try again.
            print("WARNING: Pickle path does not exist. Creating... If this message persists, something is wrong")
            os.makedirs(os.path.dirname(pickle_filename))
            f = open(pickle_filename, 'wb')
        f.close()
        compress_pickle.dump(df_to_save, pickle_filename)


if __name__ == "__main__":
    pp = Preprocessor()
    pp.assign_types_and_merge()
    pp.clean_noisy_data()
