from pyspark import SparkContext, SparkConf, SQLContext


class basic_similarity:
    """
    Very fast and lazy approach to generating similarity matrices
    while using pyspark. Depending on the time available, may not
    be the final implementation of similarity.
    """

    def __init__(self):
        self.conf = SparkConf()
        self.conf.set("spark.driver.memory", "10g")
        self.conf.set("spark.cores.max", "4")
        self.conf.set("spark.executor.heartbeatInterval", "3600")
        self.conf.setAppName("word2vec")

        self.spark_context = SparkContext.getOrCreate(self.conf)
        self.spark = SQLContext(self.spark_context)

    def generate_movie_similarity(self):
        """
        Generates embedded vectors for the actor and genres,
        appends them to one another and computes cosine similarity
        :return:
        """
        pass

    def generate_user_similarity(self):
        """
        Averages the vectors of the movies they've seen and
        computes the similarity matrix
        :return:
        """
        pass
