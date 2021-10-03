import numpy as np
import pandas as pd
import pyspark
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.ml.linalg import Vectors, VectorUDT, SparseVector
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, ArrayType
from pyspark.sql.functions import udf

from useful_tools import pickle_manager

test = 0

class VectorEmbedder:
    """
    Functor. Embeds a given column of a pyspark dataframe
    """

    def __init__(self, attr_col, attr_col_name):
        """
        First step is to generate a map of the string to the key of it
        :param attr_col: The column to use
        :param attr_col_name: The name of the column
        """
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

    def generate_embedded_vector(self, value_string):
        """
        Generates an embedded vector for the given string
        """
        vector = [0] * self.length
        for value in value_string.split(sep=", "):
            vector[self.class_dict[value]] = 1
        return vector


def to_sparse(array_type):
    array_type = np.array(array_type)
    nonzero = np.nonzero(array_type)[0]
    return Vectors.sparse(len(array_type), nonzero, array_type[nonzero])


def embed_vector(df, to_embed, embedded_column):
    """
    Uses the VectorEmbedder to embed a vector in a given dataframe
    :param df:
    :param to_embed:
    :param embedded_column:
    :return:
    """
    print("There are", df.select(to_embed).count(), "things to embed for", to_embed)
    setup = VectorEmbedder(df.select(to_embed).collect(), to_embed)
    print("That means that setup's length is", setup.length)
    generated_embedded_vector_udf = udf(
        lambda x: to_sparse(setup.generate_embedded_vector(x)),
        VectorUDT()
    )
    return df.withColumn(embedded_column, generated_embedded_vector_udf(to_embed))


def calculate_gower_distance(row, row2, ranges):
    sum_sj = 0
    for col in range(len(row)):
        if type(row[col]) == SparseVector:
            # Embedded vector - Need to go through and generate NTT/NNEQ/NFF
            # See article
            ntt = 0
            nneq = 0
            nff = 0
            for j in range(len(row[col])):
                if row[col][j] and row2[col][j]:
                    ntt += 1
                elif row[col][j] != row2[col][j]:
                    nneq += 1
                else:
                    nff += 1
            sum_sj += nneq / (ntt + (nneq + ntt))
        else:
            # Numerical
            sum_sj += 1 - (np.abs(row[col] - row2[col]) / ranges[col])

    return 1 - (sum_sj / len(row))

class GowerFunctor:
    def __init__(self, row, ranges):
        self.row = row
        self.ranges = ranges



class SimilarityRowGenerator:
    """
    Generates a single similarity row
    """

    def __init__(self, subject_movie, df: pyspark.sql.DataFrame):
        """
        :param subject_movie: The movie to base the row off of
        :param df: The entire dataframe, must contain subject_movie as a column
        """
        self.subject_movie = subject_movie

        self.ranges = {}
        idx = 0
        for col in df.columns:
            if df.select(col).dtypes[0][1] == 'int' or df.select(col).dtypes[0][1] == 'float':
                max = df.agg({col: "max"}).collect()[0][0]
                min = df.agg({col: "min"}).collect()[0][0]
                self.ranges[idx] = max - min
            else:
                self.ranges[idx] = 0
            idx += 1

        # self.sim_matrix = np.zeros((df.count(), df.count()), dtype=np.float16)

    def generate_movie_similarity(self, df):
        """
        Generates embedded vectors for the actor and genres,
        appends them to one another, computes cosine similarity
        and saves the matrix
        :return:
        """
        # TODO: BROKEN
        df.show(10)
        df.printSchema()

        df = df.rdd.map(lambda x: calculate_gower_distance(x, self.subject_movie, self.ranges))

        print(df.collect())
        return df.collect()

    def generate_user_similarity(self):
        """
        Averages the vectors of the movies they've seen and
        computes the similarity matrix
        :return:
        """
        pass


def get_ranges(df):
    ranges = {}
    idx = 0
    for col in df.columns:
        if df.select(col).dtypes[0][1] == 'int' or df.select(col).dtypes[0][1] == 'float':
            max = df.agg({col: "max"}).collect()[0][0]
            min = df.agg({col: "min"}).collect()[0][0]
            ranges[idx] = max - min
        else:
            ranges[idx] = 0
        idx += 1

    return ranges


def init_structtype():
    return StructType([
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


def drop_useless_columns(df):
    df = df.drop("imdbId") \
        .drop("title") \
        .drop("original_title") \
        .drop("date_published") \
        .drop("genre") \
        .drop("country") \
        .drop("language") \
        .drop("director") \
        .drop("writer") \
        .drop("production_company") \
        .drop("actors") \
        .drop("description") \
        .drop("usa_gross_income") \
        .drop("worldwide_gross_income") \
        .drop("metascore") \
        .drop("reviews_from_users") \
        .drop("reviews_from_critics")
    return df


def generate_movie_similarity_one_row(subject_movie, df, ranges):
        """
        Generates embedded vectors for the actor and genres,
        appends them to one another, computes cosine similarity
        and saves the matrix
        :return:
        """
        print("Getting new movie similarity list")
        new_df = df.rdd.map(lambda x: calculate_gower_distance(x, subject_movie, ranges))

        collected_rdd = new_df.collect()

        print(collected_rdd)
        return collected_rdd


def generate_movie_similarity_all(df, ranges):

    sim_matrix = []

    collected_rdd = df.rdd.collect()

    for row in collected_rdd:
        sim_matrix.append(generate_movie_similarity_one_row(row, df, ranges))

    return sim_matrix


# Tester code
if __name__ == "__main__":
    # ================= init spark =================
    conf = SparkConf()
    conf.set("spark.driver.memory", "16g")
    conf.set("spark.executor.memory", "8g")
    conf.set("spark.driver.maxResultSize", "0")
    conf.set("spark.cores.max", "12")
    conf.set("spark.executor.heartbeatInterval", "3600")

    sc = SparkContext.getOrCreate(conf)
    sc.setLogLevel('ERROR')

    spark = SQLContext(sc)
    print("Completed initialisation. Don't worry about the previous error messages")

    movie_df: pd.DataFrame = pd.DataFrame(pickle_manager.load_pickle("../pickles/organised_movies.pickle.lz4"))

    # Shuffle it so that we're being fair
    movie_df = movie_df.dropna().sample(2000, random_state=42)
    pickle_manager.save_lzma_pickle(movie_df, "movie_df_in_pandas_form.pickle.lz4")

    movies_that_exist = set()
    for index, row in movie_df.iterrows():
        movies_that_exist.add(row['imdbId'])

    pickle_manager.save_lzma_pickle(movies_that_exist, "movies_that_exist.pickle.lz4")

    rating_df: pd.DataFrame = pd.DataFrame(pickle_manager.load_pickle("../pickles/organised_ratings.pickle.lz4"))
    to_delete = []
    for index, row in rating_df.iterrows():
        if row['imdbId'] not in movies_that_exist:
            to_delete.append(index)

    rating_df = rating_df.drop(to_delete)

    pickle_manager.save_lzma_pickle(rating_df, "rating_df.pickle.lz4")

    # ================= init dataset =================
    # movie_df = pickle_manager.load_pickle("../pickles/organised_movies.pickle.lz4")
    schema = init_structtype()

    movie_df = spark.createDataFrame(movie_df, schema=schema).dropna()
    movie_df = movie_df.repartition(12)

    # ================= embed sets =================
    movie_df = embed_vector(
        movie_df, "genre", "genre_feature"
    )

    movie_df = embed_vector(
        movie_df, "actors", "actor_feature"
    )

    # ================= clean out dataset =================
    small_df = drop_useless_columns(movie_df)
    small_df.printSchema()
    small_df.show()

    # ================= build similarity matrix and save =================
    # sim = SimilarityRowGenerator(small_df.rdd.first(), small_df)



    # sim.generate_movie_similarity(small_df)
    # small_df = small_df.foreach(lambda x: generate_movie_similarity_all(x, broadcast_df, get_ranges(broadcast_df), sim_matrix, idx))
    sim_matrix = generate_movie_similarity_all(small_df, get_ranges(small_df))
    # TODO: Merge rows into a single similarity matrix which has [a][b] operators
    print(sim_matrix)
    pickle_manager.save_lzma_pickle(sim_matrix, "sim_matrix.pickle.lz4")
