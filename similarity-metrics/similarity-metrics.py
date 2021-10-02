import numpy as np
import pyspark
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.sql.types import StructType,StructField, StringType, IntegerType, FloatType, ArrayType
from pyspark.sql.functions import udf


class VectorSetup:
    """
    Sets up vector for a specific feature. Pass feature column for whole dataset to constructor.
    """
    def __init__(self, attr_col, attr_col_name):
        self.col_name = attr_col_name
        self.class_dict = {}
        idx = 0
        for row in attr_col:
            for item in row[0].split(sep=", "):
                try:
                    existing_idx = self.class_dict[item]
                except KeyError:
                    self.class_dict[item] = idx
                    idx += 1

        self.length = idx

    def compute_vector(self, value_string):
        """Run against all rows in dataframe (using map)."""
        vector = [0] * self.length
        for value in value_string.split(sep=", "):
            vector[self.class_dict[value]] = 1

        return vector


class BasicSimilarity:
    """
    Very fast and lazy approach to generating similarity matrices
    while using pyspark. Depending on the time available, may not
    be the final implementation of similarity.
    """

    def __init__(self, spark: SQLContext, subject_movie, df: pyspark.sql.DataFrame):
        self.spark: SQLContext = spark
        self.subject_movie = subject_movie

        self.ranges = {}
        for col in df.columns:
            self.ranges[col] = df.agg({col: "np.ptp"}).collect()

    def gower_distance(self, df_row: pyspark.sql.Row):
        # https://medium.com/analytics-vidhya/gowers-distance-899f9c4bd553
        sum_sj = 0
        for col in df_row.columns:
            if type(df_row[col]) == np.ndarray:
                # Categorical
                ntt = 0
                nneq = 0
                nff = 0
                for j in range(df_row[col]):
                    # Iterate through array and get DiceDistance
                    if df_row[col][j] and self.subject_movie[col][j]:
                        ntt += 1
                    elif df_row[col][j] != self.subject_movie[col][j]:
                        nneq += 1
                    else:
                        nff += 1
                sum_sj += nneq / (ntt + (nneq + ntt))
            else:
                # Numerical
                sum_sj += 1 - (np.abs(df_row[col] - self.subject_movie[col]) / self.ranges[col])

        return 1 - (sum_sj / len(df_row.columns))

    def generate_movie_similarity(self, df):
        """
        Generates embedded vectors for the actor and genres,
        appends them to one another and computes cosine similarity
        :return:
        """

        gower_distances = sorted(df.rdd.map(self.gower_distance).collect())


    def generate_user_similarity(self):
        """
        Averages the vectors of the movies they've seen and
        computes the similarity matrix
        :return:
        """
        pass


from useful_tools import pickle_manager


# Tester code
if __name__ == "__main__":
    movie_df = pickle_manager.load_pickle("../pickles/organised_movies.pickle.lz4")

    conf = SparkConf()
    conf.set("spark.driver.memory", "10g")
    conf.set("spark.cores.max", "4")
    conf.set("spark.executor.heartbeatInterval", "3600")

    sc = SparkContext.getOrCreate(conf)

    spark = SQLContext(sc)

    schema = StructType([
        StructField("imdbId", StringType(), False),
        StructField("title", StringType(), True),
        StructField("original_title", StringType(), True),
        StructField("date_published", StringType(), True),
        StructField("genre", StringType(), True),
        StructField("duration", IntegerType(), True),
        StructField("country", StringType(), True),
        StructField("language", StringType(), True),
        StructField("director", StringType(), True),
        StructField("writer", StringType(), True),
        StructField("production_company", StringType(), True),
        StructField("actors", StringType(), True),
        StructField("description", StringType(), True),
        StructField("avg_vote", FloatType(), True),
        StructField("votes", IntegerType(), True),
        StructField("usa_gross_income", StringType(), True),
        StructField("worldwide_gross_income", StringType(), True),
        StructField("budget", FloatType(), True),
        StructField("year", IntegerType(), True),
        StructField("metascore", IntegerType(), True),
        StructField("reviews_from_users", IntegerType(), True),
        StructField("reviews_from_critics", IntegerType(), True)
    ])

    movie_df = spark.createDataFrame(movie_df, schema=schema)

    genre_setup = VectorSetup(movie_df.select("genre").collect(), "genre")
    test_udf = udf(lambda x: genre_setup.compute_vector(x), ArrayType(IntegerType()))
    movie_df = movie_df.withColumn("genre_fixed", test_udf("genre"))
    movie_df.show(10)

    # TODO: Can convert string to vector -> just need to test similarity finding now