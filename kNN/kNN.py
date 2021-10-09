import typing
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from useful_tools import pickle_manager
from collections import namedtuple
from pycaret import classification

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
            per_user_ratings = self.rating_df[self.rating_df["userId"] == user]

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
        similarities = zip(self.sim_matrix[sim_index], [i for i in range(len(self.sim_matrix[sim_index]))])
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

        accuracy_list = []
        tested_movies_list = []
        for user_id in self.user_ratings.keys():
            if len(self.user_ratings[user_id].training) == 0 or len(self.user_ratings[user_id].testing) == 0:
                continue
            accuracy = self._get_user_accuracy(user_id)
            accuracy_list.append(accuracy)
            tested_movies_list.append(len(self.user_ratings[user_id].testing))
            sum_accuracy += accuracy
            num += 1
            print(f"Accuracy for user_id {user_id}: {accuracy}. Average accuracy so far: {sum_accuracy / num}")

        return sum_accuracy / num, accuracy_list, tested_movies_list


def clean_sim_matrix(sim_matrix):
    for i in range(len(sim_matrix)):
        sim_matrix[i][i] = 0

    return sim_matrix


class SciKNN:
    def __init__(self, user_ratings, movie_df: pd.DataFrame, k=6):
        self.user_ratings = user_ratings
        self.movie_df: pd.DataFrame = movie_df
        self.k = k

    def _prep_for_user(self, user_id):

        le = preprocessing.LabelEncoder()

        genre = []
        duration = []
        director = []
        writer = []
        production_company = []
        description = []
        actors = []
        avg_vote = []
        budget = []
        year = []

        train_classes = []

        for index, row in self.user_ratings[user_id].training.iterrows():
            movie_row_idx = np.where(self.movie_df["imdbId"] == row["imdbId"])
            movie_row = self.movie_df.iloc[movie_row_idx[0][0]]

            genre.append(movie_row["genre"])
            duration.append(movie_row["duration"])
            director.append(movie_row["director"])
            writer.append(movie_row["writer"])
            production_company.append(movie_row["production_company"])
            description.append(movie_row["description"])
            actors.append(movie_row["actors"])
            avg_vote.append(movie_row["avg_vote"])
            budget.append(movie_row["budget"])
            year.append(movie_row["year"])

            train_classes.append(row["class"])

        genre = le.fit_transform(genre)
        director = le.fit_transform(director)
        writer = le.fit_transform(writer)
        production_company = le.fit_transform(production_company)
        description = le.fit_transform(description)
        actors = le.fit_transform(actors)

        train_features = list(zip(genre, duration, director, writer, production_company, description, actors, avg_vote, budget, year))

        genre = []
        duration = []
        director = []
        writer = []
        production_company = []
        description = []
        actors = []
        avg_vote = []
        budget = []
        year = []

        test_classes = []

        for index, row in self.user_ratings[user_id].testing.iterrows():
            movie_row_idx = np.where(self.movie_df["imdbId"] == row["imdbId"])
            movie_row = self.movie_df.iloc[movie_row_idx[0][0]]

            genre.append(movie_row["genre"])
            duration.append(movie_row["duration"])
            director.append(movie_row["director"])
            writer.append(movie_row["writer"])
            production_company.append(movie_row["production_company"])
            description.append(movie_row["description"])
            actors.append(movie_row["actors"])
            avg_vote.append(movie_row["avg_vote"])
            budget.append(movie_row["budget"])
            year.append(movie_row["year"])

            test_classes.append(row["class"])

        genre = le.fit_transform(genre)
        director = le.fit_transform(director)
        writer = le.fit_transform(writer)
        production_company = le.fit_transform(production_company)
        description = le.fit_transform(description)
        actors = le.fit_transform(actors)

        test_features = list(
            zip(genre, duration, director, writer, production_company, description, actors, avg_vote, budget, year))

        return train_features, train_classes, test_features, test_classes

    def _run_knn_for_user(self, user_id):
        train_features, train_classes, test_features, test_classes = self._prep_for_user(user_id)

        k = min(len(train_classes), self.k)
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(train_features, train_classes)

        predicted = model.predict(test_features)

        correct = 0
        incorrect = 0
        for i in range(len(predicted)):
            if predicted[i] == test_classes[i]:
                correct += 1
            else:
                incorrect += 1

        return correct / (correct + incorrect)

    def evaluate(self):
        num = 0
        sum_accuracy = 0

        accuracy_list = []
        tested_movies_list = []
        for user_id in self.user_ratings.keys():
            if len(self.user_ratings[user_id].training) == 0 or len(self.user_ratings[user_id].testing) == 0:
                continue
            accuracy = self._run_knn_for_user(user_id)
            accuracy_list.append(accuracy)
            tested_movies_list.append(len(self.user_ratings[user_id].testing))
            sum_accuracy += accuracy
            num += 1
            print(f"Accuracy for user_id {user_id}: {accuracy}. Average accuracy so far: {sum_accuracy / num}")

        return sum_accuracy / num, accuracy_list, tested_movies_list


class PycaretKNN:
    def __init__(self, user_ratings, movie_df: pd.DataFrame, k=6):
        self.user_ratings = user_ratings
        self.movie_df: pd.DataFrame = movie_df
        self.k = k

    @staticmethod
    def _add_class_to_movie_df(row, training, testing):
        if row["imdbId"] in testing.imdbId.unique():
            idx = np.where(testing["imdbId"] == row["imdbId"])
            return testing.iloc[idx[0][0]]["class"]
        else:
            idx = np.where(training["imdbId"] == row["imdbId"])
            return training.iloc[idx[0][0]]["class"]

    def _format_for_pycaret(self, user_id):
        training_ids = self.user_ratings[user_id].training.imdbId.unique()
        testing_ids = self.user_ratings[user_id].testing.imdbId.unique()
        to_delete = []
        for index, row in self.movie_df.iterrows():
            if row["imdbId"] not in training_ids and row["imdbId"] not in testing_ids:
                to_delete.append(index)

        user_movie_df = self.movie_df.drop(to_delete)
        user_movie_df.insert(len(user_movie_df.columns), "class", user_movie_df.apply(lambda row: self._add_class_to_movie_df(row, self.user_ratings[user_id].training, self.user_ratings[user_id].testing), axis=1))

        return user_movie_df

    def _get_accuracy_for_user(self, user_id):
        user_df = self._format_for_pycaret(user_id)

        k = min(self.k, int(len(user_df) / 4))

        user_setup = classification.setup(data=user_df, target='class', session_id=678, silent=True)

        knn = classification.create_model('knn', n_neighbors=k, fold=2)
        prediction = classification.predict_model(knn)

        correct = 0
        incorrect = 0
        for index, row in prediction.iterrows():
            if row["class"] == row["Label"]:
                correct += 1
            else:
                incorrect += 1

        return correct / (correct + incorrect), correct + incorrect

    def evaluate(self):
        num = 0
        sum_accuracy = 0

        accuracy_list = []
        tested_movies_list = []
        for user_id in self.user_ratings.keys():
            try:
                accuracy, tested_movies = self._get_accuracy_for_user(user_id)
                sum_accuracy += accuracy
                num += 1
                accuracy_list.append(accuracy)
                tested_movies_list.append(tested_movies)
                print(f"Accuracy for user_id {user_id}: {accuracy}. Average accuracy so far: {sum_accuracy / num}")
            except:
                print(f"Failed to get accuracy for user_id {user_id}, skipping")
                continue

        return sum_accuracy / num, accuracy_list, tested_movies_list


def main():
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

    original_output = []
    for k in range(1, 6):
        knn_classifier = OriginalKNN(sim_matrix, rating_sets, movies_that_exist, movies_df, k=k)
        accuracy, accuracy_list, tested_movies_list = knn_classifier.evaluate()
        original_output.append((accuracy, accuracy_list, tested_movies_list))
        print(f"Original kNN accuracy at k={k}: {accuracy}")
    pickle_manager.save_lzma_pickle(original_output, "../pickles/original_knn_output.pickle.lz4")

    sciknn_output = []
    for k in range(1, 6):
        knn_classifier = SciKNN(rating_sets, movies_df, k=k)
        accuracy, accuracy_list, tested_movies_list = knn_classifier.evaluate()
        sciknn_output.append((accuracy, accuracy_list, tested_movies_list))
        print(f"Sk-learn kNN accuracy at k={k}: {accuracy}")
    pickle_manager.save_lzma_pickle(sciknn_output, "../pickles/sklearn_knn_output.pickle.lz4")

    pycaretknn_output = []
    for k in range(1, 6):
        knn_classifier = PycaretKNN(rating_sets, movies_df, k=k)
        accuracy, accuracy_list, tested_movies_list = knn_classifier.evaluate()
        pycaretknn_output.append((accuracy, accuracy_list, tested_movies_list))
        print(f"PyCaret kNN accuracy at k={k}: {accuracy}")
    pickle_manager.save_lzma_pickle(pycaretknn_output, "../pickles/pycaret_knn_output.pickle.lz4")


if __name__ == "__main__":
    main()
