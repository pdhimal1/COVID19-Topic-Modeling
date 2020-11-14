"""
Data from

COVID-19 Open Research Dataset Challenge (CORD-19)
https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge

Prakash Dhimal
Manav Garkel
George Mason University
CS 657 Mining Massive Datasets
Assignment 3: Topic Modeling

Examples:

TODO - delete these when done

COVID EDA: Initial Exploration Tool - done
https://www.kaggle.com/ivanegapratama/covid-eda-initial-exploration-tool

Topic modeling examples:
https://www.kaggle.com/danielwolffram/topic-modeling-finding-related-articles

Literature clustering:
https://www.kaggle.com/maksimeren/covid-19-literature-clustering

Reading this data with pyspark - done
https://www.kaggle.com/jonathanbesomi/cord-19-sources-unification-with-pyspark-sql
"""

import glob

from pyspark.ml.feature import CountVectorizer, Tokenizer
from pyspark.mllib.clustering import LDA
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.sql.types import StringType


def init_spark():
    spark = SparkSession.builder.appName("HW3-data-exploration").getOrCreate()
    return spark


def read_metadata(root_path, spark):
    metadata_path = f'{root_path}/metadata.csv'
    meta_df = spark.read.option("header", True).csv(metadata_path)
    return meta_df


def read_json_files(root_path, spark):
    all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
    # todo - for now restrict this to 100 files
    all_json = all_json[:100]

    data = spark.read.json(all_json, multiLine=True)
    data.createOrReplaceTempView("data")
    return data


def get_title_abstract_text(spark, data):
    # Select text columns
    # todo - add more columns
    covid_sql = spark.sql(
        """
            SELECT
                body_text.text AS body_text,
                paper_id
            FROM data
            """)
    return covid_sql


def parseVectors(line):
    return [int(line[1]), line[0]]


def main():
    root_path = '../data/archive/'
    spark = init_spark()
    json_files = read_json_files(root_path, spark)
    data = get_title_abstract_text(spark, json_files)

    word_join_f = F.udf(lambda x: [''.join(w) for w in x], StringType())
    data_df = data.withColumn("body_text", word_join_f("body_text"))

    # Tokenize the text in the text column
    tokenizer = Tokenizer(inputCol="body_text", outputCol="words")
    wordsDataFrame = tokenizer.transform(data_df)

    # remove 20 most occuring documents, documents with non numeric characters, and documents with <= 3 characters
    cv_tmp = CountVectorizer(inputCol="words", outputCol="vectors")
    cvmodel = cv_tmp.fit(wordsDataFrame)
    df_vect = cvmodel.transform(wordsDataFrame)

    # todo - need to add id column here?
    df_vect = df_vect.select("*").withColumn("id", monotonically_increasing_id())

    # sparsevector = df_vect.select('vectors', 'id').rdd.map(parseVectors)
    # todo - use the parseVectors method above
    sparsevector = df_vect.select('vectors').rdd.map(list)

    # n_components=50, random_state=0
    num_topics = 10
    max_iterations = 50
    # Train the LDA model, set seed?
    model = LDA.train(sparsevector, k=num_topics, maxIterations=max_iterations)

    # Print the topics in the model
    # todo - make this better, use viz stuff here.
    topics = model.describeTopics(maxTermsPerTopic=15)
    for x, topic in enumerate(topics):
        print('topic nr: ' + str(x))
        words = topic[0]
        weights = topic[1]
        for n in range(len(words)):
            print(cvmodel.vocabulary[words[n]] + ' ' + str(weights[n]))


if __name__ == '__main__':
    main()
