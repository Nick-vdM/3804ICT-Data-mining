import numpy as np
import pyspark
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.ml.linalg import Vectors, VectorUDT
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, ArrayType
from pyspark.sql.functions import udf

from useful_tools import pickle_manager


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


def embed_vector(df, to_embed, embedded_column):
    """
    Uses the VectorEmbedder to embed a vector in a given dataframe
    :param df:
    :param to_embed:
    :param embedded_column:
    :return:
    """
    setup = VectorEmbedder(df.select(to_embed).collect(), to_embed)
    generated_embedded_vector_udf = udf(
        lambda x: Vectors.dense(setup.generate_embedded_vector(x)),
        VectorUDT()
    )
    return df.withColumn(embedded_column, generated_embedded_vector_udf(to_embed))


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
        for col in df.columns:
            if df.select(col).dtypes[0][0] == 'int':
                max = df.agg({col: "max"}).collect()[0][0]
                min = df.agg({col: "min"}).collect()[0][0]
                self.ranges[col] = max - min
            else:
                self.ranges[col] = 0

        # self.sim_matrix = np.zeros((df.count(), df.count()), dtype=np.float16)

    def calculate_gower_distance(self, df_row: pyspark.sql.Row):
        """
        Due to how our columns have a different impact on the similarity
        between movies, we need to account for a different weight between each
        one. See https://medium.com/analytics-vidhya/gowers-distance-899f9c4bd553
        :param df_row: The row to use
        :example
        df.rdd.map(self.calculate_gower_distance)
        :return:
        """
        sum_sj = 0
        for col in df_row.columns:
            if type(df_row[col]) == ArrayType(IntegerType()):
                # Embedded vector - Need to go through and generate NTT/NNEQ/NFF
                # See article
                ntt = 0
                nneq = 0
                nff = 0
                for j in range(df_row[col]):
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
        appends them to one another, computes cosine similarity
        and saves the matrix
        :return:
        """
        # TODO: BROKEN
        df.show()
        df.printSchema()
        print("generating rdd")
        rdd = df.rdd
        print("Trying to print collect")
        print("The rdd is", rdd.count(), "long")
        print(rdd.collect())
        print("mapping rdd")
        rdd2 = rdd.map(lambda x: (x, 1))
        print("trying to print rdd")
        for element in rdd2.collect():
            print(element)

        gower_distances = rdd.map(self.calculate_gower_distance)
        gower_distances.collect()

    def generate_user_similarity(self):
        """
        Averages the vectors of the movies they've seen and
        computes the similarity matrix
        :return:
        """
        pass


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


# Tester code
if __name__ == "__main__":
    # ================= init spark =================
    conf = SparkConf()
    conf.set("spark.driver.memory", "16g")
    conf.set("spark.executor.memory", "8g")
    conf.set("spark.driver.maxResultSize", "0")
    conf.set("spark.cores.max", "4")
    conf.set("spark.executor.heartbeatInterval", "3600")

    sc = SparkContext.getOrCreate(conf)
    sc.setLogLevel('ERROR')

    spark = SQLContext(sc)

    # ================= init dataset =================
    movie_df = pickle_manager.load_pickle("../pickles/organised_movies.pickle.lz4")
    schema = init_structtype()

    movie_df = spark.createDataFrame(movie_df, schema=schema)

    # ================= embed sets =================
    movie_df = embed_vector(
        movie_df, "genre", "genre_feature"
    )

    movie_df = embed_vector(
        movie_df, "actors", "actor_feature"
    )

    # ================= clean out dataset =================
    small_df = drop_useless_columns(movie_df)

    print(small_df.rdd.collect())
    print("Printed rdd")
    exit(1)

    small_df.show()

    # TODO: BROKEN
    print("generating rdd")

    rdd = small_df.rdd
    print("Trying to print collect")
    print("The rdd is", rdd.count(), "long")
    print(rdd.collect())
    # ================= build similarity matrix and save =================
    sim = SimilarityRowGenerator(small_df[0], small_df)
    sim.generate_movie_similarity(small_df)
    # TODO: Merge rows into a single similarity matrix which has [a][b] operators
