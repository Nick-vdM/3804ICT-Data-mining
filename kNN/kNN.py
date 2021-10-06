import typing

import numpy as np
import pandas as pd
from useful_tools import pickle_manager
from collections import namedtuple

TrainTest = namedtuple('TrainTest', 'training testing')


class SetSeparator:
    def __init__(self, rating_df: pd.DataFrame):
        self.rating_df: pd.DataFrame = rating_df
        self.user_ids = rating_df.userId.unique()
        self._add_classes_to_df()

    def _add_classes_to_df(self):
        self.rating_df.insert(len(self.rating_df.columns), "class", self.rating_df.apply(lambda row: self._convert_num_to_class(row), axis=1))

    @staticmethod
    def _convert_num_to_class(row):
        if row["rating"] < 3:
            return "Dislike"
        elif row["rating"] > 3:
            return "Like"
        else:
            return "Neutral"

    def get_sets_for_all_users(self, test_proportion=0.2):
        user_ratings = {}
        for user in self.user_ids:
            per_user_ratings = rating_df[rating_df["userId"] == user]

            train_set = per_user_ratings.sample(frac=(1-test_proportion), random_state=789)
            user_ratings[user] = TrainTest(train_set, per_user_ratings.drop(train_set.index))

        return user_ratings


class OriginalKNN:
    def __init__(self, sim_matrix, user_ratings, movies_that_exist, movie_df, k=6):
        self.sim_matrix = sim_matrix
        self.user_ratings: typing.Dict[int, TrainTest] = user_ratings
        self.movies_that_exist = movies_that_exist
        self.movie_df: pd.DataFrame = movie_df
        self.k = k

    def _get_class(self, user_id, movie_id):
        training = self.user_ratings[user_id].training

        sim_index = np.where(self.movie_df["imdbId"] == movie_id)[0][0]
        similarities = zip(self.sim_matrix[sim_index], [i for i in range(len(sim_matrix[sim_index]))])
        sorted_similarities = sorted(similarities, key=lambda x: x[0], reverse=True)

        like = 0
        neutral = 0
        dislike = 0
        for sim in sorted_similarities:
            if like+neutral+dislike >= self.k:
                break
            imdb_id = self.movie_df.iloc[sim[1]]["imdbId"]

            index_of_rating = np.where(training["imdbId"] == imdb_id)
            if index_of_rating[0].size != 1:
                continue
            else:
                classification = training.iloc[index_of_rating[0][0]]["class"]
                if classification == "Like":
                    like += 1
                elif classification == "Dislike":
                    dislike += 1
                else:
                    neutral += 1

        # If clear winner
        if like > neutral and like > dislike:
            # If like winner
            return "Like"
        elif neutral > like and neutral > dislike:
            # If neutral winner
            return "Neutral"
        elif dislike > like and dislike > neutral:
            # If dislike winner
            return "Dislike"

        # If two way tie
        if like == neutral and like != dislike:
            # If like and neutral equal
            return "Neutral"
        elif like == dislike and like != neutral:
            # If like and dislike equal
            return "Dislike"
        elif neutral == dislike and neutral != like:
            # If neutral and dislike equal
            return "Neutral"

        # If all equal return neutral
        return "Neutral"

    def _get_user_accuracy(self, user_id):

        correct = 0
        incorrect = 0
        test_set = self.user_ratings[user_id].testing

        for index, row in test_set.iterrows():
            classification = self._get_class(user_id, row["imdbId"])
            if classification == row["class"]:
                correct += 1
            else:
                incorrect += 1

        return correct / (correct + incorrect)

    def evaluate(self):
        num = 0
        sum_accuracy = 0

        for user_id in self.user_ratings.keys():
            if len(self.user_ratings[user_id].training) == 0 or len(self.user_ratings[user_id].testing) == 0:
                continue
            sum_accuracy += self._get_user_accuracy(user_id)
            num += 1
            print(f"Accuracy for user_id {user_id}: {sum_accuracy / num}")

        return sum_accuracy / num


def clean_sim_matrix(sim_matrix):
    for i in range(len(sim_matrix)):
        sim_matrix[i][i] = 0

    return sim_matrix


if __name__ == "__main__":
    rating_df: pd.DataFrame = pickle_manager.load_pickle("../pickles/rating_df.pickle.lz4")
    movies_that_exist: set = pickle_manager.load_pickle("../pickles/movies_that_exist.pickle.lz4")
    movies_df: pd.DataFrame = pickle_manager.load_pickle("../pickles/movie_df_in_pandas_form.pickle.lz4")
    sim_matrix: pd.DataFrame = pickle_manager.load_pickle("../pickles/sim_matrix.pickle.lz4")

    sim_matrix = clean_sim_matrix(sim_matrix)

    print(rating_df.head(10))
    print(movies_that_exist)
    print(movies_df.head(10))
    print(sim_matrix[1])

    set_sep = SetSeparator(rating_df)
    rating_sets = set_sep.get_sets_for_all_users()

    knn_classifier = OriginalKNN(sim_matrix, rating_sets, movies_that_exist, movies_df)

    print(f"Accuracy at k=6: {knn_classifier.evaluate()}")