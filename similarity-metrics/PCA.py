"""
Takes the movie_df_in_pandas_form.pickle.lz4 and runs through all of the steps
again
"""
from useful_tools import pickle_manager
from sklearn.decomposition import PCA

sim_metrics = __import__('similarity-metrics')
import pandas as pd

import pandas as pd
import pyspark
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.ml.linalg import Vectors, VectorUDT, SparseVector
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, ArrayType
from pyspark.sql.functions import udf

from useful_tools import pickle_manager


def catch_up():
    print("\n\n\n\n\nCompleted initialisation. Don't worry about the previous error messages")
    movie_df = pickle_manager.load_pickle(
        '../pickles/movie_df_in_pandas_form.pickle.lz4'
    )

    movies_that_exist = set()
    for index, row in movie_df.iterrows():
        movies_that_exist.add(row['imdbId'])

    pickle_manager.save_lzma_pickle(
        movies_that_exist,
        "../pickles/movies_that_exist.pickle.lz4"
    )

    rating_df: pd.DataFrame = pd.DataFrame(pickle_manager.load_pickle("../pickles/organised_ratings.pickle.lz4"))
    to_delete = []
    for index, row in rating_df.iterrows():
        if row['imdbId'] not in movies_that_exist:
            to_delete.append(index)

    rating_df = rating_df.drop(to_delete)

    pickle_manager.save_lzma_pickle(
        rating_df, "../pickles/rating_df.pickle.lz4"
    )

    # ================= init dataset =================
    # movie_df = pickle_manager.load_pickle("../pickles/organised_movies.pickle.lz4")
    schema = sim_metrics.init_structtype()

    movie_df = spark.createDataFrame(movie_df, schema=schema).dropna()
    movie_df = movie_df.repartition(12)

    # ================= embed sets =================
    movie_df = sim_metrics.embed_vector(
        movie_df, "genre", "genre_feature"
    )

    movie_df = sim_metrics.embed_vector(
        movie_df, "actors", "actor_feature"
    )

    # ================= clean out dataset =================
    small_df = sim_metrics.drop_useless_columns(movie_df)
    small_df.printSchema()
    # =====================================================
    return small_df


def convert_to_pandas_df(small_df):
    """
    :param small_df: a spark df that contains a genre and actors feature column
    that are made out of vectors
    :return:
    """
    pandas_df = small_df.toPandas()

    vec2array = lambda x: x.toArray()
    pandas_df['actor_feature'] = pandas_df['actor_feature'].apply(
        vec2array
    )
    pandas_df['genre_feature'] = pandas_df['genre_feature'].apply(
        vec2array
    )
    return pandas_df


if __name__ == "__main__":
    # ================= init spark =================
    conf = SparkConf()
    conf.set("spark.driver.memory", "16g")
    conf.set("spark.executor.memory", "16g")
    conf.set("spark.driver.maxResultSize", "0")
    conf.set("spark.cores.max", "6")
    conf.set("spark.executor.cores", "6")
    conf.set("spark.executor.heartbeatInterval", "3600")

    sc = SparkContext.getOrCreate(conf)
    sc.setLogLevel('ERROR')
    spark = SQLContext(sc)

    small_df = catch_up()
    pandas_df = convert_to_pandas_df(small_df)

    print(pandas_df.values)
    print(pandas_df.dtypes)
    print(pandas_df.head(10))
