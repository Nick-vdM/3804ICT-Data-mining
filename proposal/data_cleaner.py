"""
To be ran after data_integrator.py, because the tables
are such a mess afterwards that this makes things easier
"""
import pandas as pd

from proposal import data_integrator
from useful_tools import pickle_manager


class data_cleaner:
    def __init__(self):
        self.user_rating_output_filename = "../data/organised_ratings.csv"
        self.movies_output_filename = "../data/organised_movies.csv"

        self.user_rating_pickle_filename = "../pickles/organised_ratings.pickle.lz4"
        self.movies_pickle_filename = "../pickles/organised_movies.pickle.lz4"

        self.user_df = pickle_manager.load_pickle(self.user_rating_pickle_filename)
        self.movie_df = pickle_manager.load_pickle(self.movies_pickle_filename)

    def clean(self):
        """
        Removes unnecessary columns, removes the lower outliers for number of reviews,
        and removes any movies that don't actually exist anymore
        :return:
        """
        print("Removing unnecessary columns")
        print("\tbefore", self.movie_df.columns)
        self.remove_unnecessary_columns()
        print("\tafter", self.movie_df.columns)

        print("Removing lower outliers for reviews")
        print("Length of table before dropping lower outliers", len(self.movie_df))
        self.movie_df = self.remove_lower_outliers(self.movie_df, 'reviews_from_users')
        self.movie_df = self.remove_lower_outliers(self.movie_df, 'reviews_from_critics')
        self.movie_df = self.remove_lower_outliers(self.movie_df, 'votes')
        self.movie_df = self.remove_outliers(self.movie_df, 'duration')

        print("Length of table after dropping lower outliers", len(self.movie_df))

        print("Removing movies that don't exist")
        # Need to do it both ways, quite an expensive operation
        print("\tSizes before: Movies", len(self.movie_df), "Ratings:", len(self.user_df))
        self.movie_df = self._remove_unknown_movies(self.movie_df, self.user_df)
        self.user_df = self._remove_unknown_movies(self.user_df, self.movie_df)
        print("\tSizes after: Movies", len(self.movie_df), "Ratings:", len(self.user_df))

        print("Saving to csv and compressed pickles")
        data_integrator.data_integrator.save_dataframe_to_csv_and_pickle(
            self.movie_df,
            self.movies_output_filename,
            self.movies_pickle_filename
        )

    def remove_unnecessary_columns(self):
        print(self.movie_df.head())
        del self.movie_df['budget']
        del self.movie_df['usa_gross_income']
        del self.movie_df['worldwide_gross_income']
        del self.movie_df['metascore']

    @staticmethod
    def remove_lower_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        interquartile_range = Q3 - Q1
        lower = Q1 - (1.5 * interquartile_range)
        df = df[(df[column] > lower)]
        return df

    def remove_outliers(self, df, column):
        """
        Removes both sides of the outliers
        :param df:
        :param column:
        :return:
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        interquartile_range = Q3 - Q1
        lower = Q1 - (1.5 * interquartile_range)
        upper = Q3 + (1.5 * interquartile_range)
        df = df[(df[column] > lower) & (df[column] < upper) & (~df[column].isna())]
        return df

    @staticmethod
    def _remove_unknown_movies(from_df, to_df):
        """
        Checks if the from_df's movie ids exist in the other one,
        and if it doesn't, then it removes that whole row
        :param from_df:
        :param to_df:
        :return:
        """
        # Generate a set of all the unique values so we have O(1) lookup
        to_df_uniques = set(to_df['imdbId'].unique())
        rows_to_remove = []

        # Rows don't have names anymore, need the loc
        column_index = from_df.columns.get_loc('imdbId')

        for row in from_df.iterrows():
            if row[1][column_index] not in to_df_uniques:
                # row[0] should be the index
                rows_to_remove = row[0]

        return from_df.drop(rows_to_remove)


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    dc = data_cleaner()
    dc.clean()
