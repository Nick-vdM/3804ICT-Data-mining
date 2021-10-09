import time
import pyspark.sql.functions as F
import numpy as np
import pandas as pd
from pyspark import SparkContext, SparkConf, SQLContext
from pyspark.sql.functions import udf
from pyspark.ml.feature import Tokenizer, StopWordsRemover, Word2Vec
from pyspark.sql.types import *
from nltk.stem.snowball import *
from useful_tools import pickle_manager


class SentenceAverager:
    """
    Averages the sentences in a dataframe
    example
        sa = SentenceAverager(sentences, pandas_w2v)
        sentences = sa.average_sentences()
    """

    def __init__(self, sentences, pandas_w2v):
        self.sentences = sentences

        self.vector_lookup = {}
        for index, row in pandas_w2v.iterrows():
            # Words should be unique so we don't have to worry about collisions
            self.vector_lookup[row['word']] = row['vector']

        self.vector_length = len(pandas_w2v['vector'][0])

    def generate_sentence_features(self):
        series = []

        for _, row in self.sentences.iterrows():
            series.append(self.average_sentence(row['processed_description']))

        return pd.DataFrame(series)

    def average_sentence(self, sentence):
        new_vector = np.zeros(self.vector_length)
        for word in sentence:
            if word not in self.vector_lookup:
                continue  # Not sure how this happens
            new_vector += self.vector_lookup[word]
        # l1 norm
        new_vector /= np.linalg.norm(new_vector, 1)

        return new_vector.tolist()


def create_word2vec(dataframe: pd.DataFrame, column: str):
    """
    :param dataframe:
    :param column:
    :return:
    """
    clean_data = clean_text_for_w2v(dataframe, column)

    word2vec = Word2Vec(inputCol="processed_description", outputCol="vectors")
    print("Starting to generate word2vec model")
    start = time.perf_counter()
    model = word2vec.fit(clean_data)
    print("Took", time.perf_counter() - start, 'seconds')

    sentences = clean_data.select('imdbId', 'title', 'description', 'processed_description').toPandas()
    pandas_w2v = model.getVectors().toPandas()

    print(sentences.head(50))
    print("Vectors are", pandas_w2v)

    return sentences, pandas_w2v
    # model.write().overwrite().save("vec.model")


def clean_text_for_w2v(dataframe: pd.DataFrame, column: str):
    """
    Cleans the given text by transforming it into a token format,
    removing the stop words (the, to, this), stemming words
    (thinking, think, thought) and returns the clean data
    :param dataframe:
    :param column:
    :return:
    """
    dataframe = dataframe.select(
        [F.regexp_replace(col, r',|\.|&|\\|\||-|_', '').alias(col) \
         for col in dataframe.columns]
    )

    tokenizer = Tokenizer(inputCol=column, outputCol="tokenized")
    words_data = tokenizer.transform(dataframe)

    remover = StopWordsRemover(
        inputCol="tokenized", outputCol="tokenized_no_stops"
    )
    words_data = remover.transform(words_data)

    stemmer = SnowballStemmer(language='english')
    stemmer_udf = udf(
        lambda word: [stemmer.stem(token) for token in word],
        ArrayType(StringType())
    )
    clean_data = words_data.withColumn(
        'processed_description',
        stemmer_udf("tokenized_no_stops"))
    clean_data.show()

    return clean_data


def main():
    # ================= init spark =================
    conf = SparkConf()
    conf.set("spark.driver.memory", "16g")
    conf.set("spark.executor.memory", "16g")
    conf.set("spark.driver.maxResultSize", "0")
    conf.set("spark.cores.max", "4")
    conf.set("spark.executor.cores", "4")
    conf.set("spark.driver.cores", "4")
    conf.set("spark.executor.heartbeatInterval", "3600")

    sc = SparkContext.getOrCreate(conf)

    sc.setLogLevel('ERROR')

    spark = SQLContext(sc)
    print("Completed initialisation. Don't worry about the previous error messages")

    movie_df: pd.DataFrame = pd.DataFrame(
        pickle_manager.load_pickle(
            "../pickles/organised_movies.pickle.lz4"
        )[['imdbId', 'title', 'description']]
    )

    print(movie_df.columns)
    print(movie_df.head())

    schema = StructType([
        StructField("imdbId", StringType(), False),
        StructField("title", StringType(), True),
        StructField('description', StringType(), True)
    ])
    spark_movie_df = spark.createDataFrame(movie_df, schema=schema)
    sentences, pandas_w2v = create_word2vec(spark_movie_df, 'description')

    sa = SentenceAverager(sentences, pandas_w2v)
    sentence_features = sa.generate_sentence_features()
    print(sentence_features.head(10))

    pickle_manager.save_lzma_pickle(
        sentences, '../pickles/sentences.pickle.lz4'
    )
    pickle_manager.save_lzma_pickle(
        pandas_w2v, '../pickles/pandas_w2v.pickle.lz4'
    )
    pickle_manager.save_lzma_pickle(
        sentence_features, '../pickles/sentence_features.pickle.lz4'
    )
    print(sentence_features.describe())


if __name__ == '__main__':
    main()
