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
import os
from time import time

from nltk.corpus import stopwords
from pyspark.ml.clustering import LDA
from pyspark.ml.feature import CountVectorizer, Tokenizer, StopWordsRemover
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType

stop_words = set(stopwords.words('english'))


def init_spark():
    spark = SparkSession.builder.appName("HW3-Coord-data").getOrCreate()
    return spark


def read_json_files(root_path, spark):
    json_dir = root_path + "document_parses/pdf_json/"
    filenames = os.listdir(json_dir)

    all_json = [json_dir + filename for filename in filenames]
    # todo - for now restrict this to 100 files
    all_json = all_json[:100]

    data = spark.read.json(all_json, multiLine=True)
    data.createOrReplaceTempView("data")
    return data


def get_body_text(spark, data):
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


def topic_render(topic, wordNumbers, vocabArray):  # specify vector id of words to actual words
    terms = topic[1]
    result = []
    for i in range(wordNumbers):
        term = vocabArray[terms[i]]
        result.append(term)
    return result


def clean_up_sentences(sentence):
    matches = [word for word in sentence.split(' ') if word.isalnum()]
    matches = [word.lower() for word in matches]
    matches = [word for word in matches if word not in stop_words]
    matches = [word for word in matches if len(word) >= 3]
    return matches


def clean_up(document):
    cleaned = [clean_up_sentences(w) for w in document]
    joined = [' '.join(w) for w in cleaned]
    return joined


def main():
    start = time()
    root_path = '../data/archive/'
    spark = init_spark()
    json_files = read_json_files(root_path, spark)
    data = get_body_text(spark, json_files)

    # clean the data
    word_clean_up_F = F.udf(lambda x: clean_up(x), StringType())
    data = data.withColumn("body_text_cleaned", word_clean_up_F("body_text"))

    tokenizer = Tokenizer(inputCol="body_text_cleaned", outputCol="words")
    token_DataFrame = tokenizer.transform(data)

    # Remove stopwords
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    cleaned_DataFrame = remover.transform(token_DataFrame)

    # Count vectorizer
    cv_tmp = CountVectorizer(inputCol="filtered", outputCol="features")
    cvmodel = cv_tmp.fit(cleaned_DataFrame)
    df_vect = cvmodel.transform(cleaned_DataFrame)

    # Fit the LDA Model
    num_topics = 10
    max_iterations = 50
    lda = LDA(seed=1, optimizer="em", k=num_topics, maxIter=max_iterations)
    lda_model = lda.fit(df_vect)
    print("done fitting")
    # joblib.dump(lda_model, 'lda.csv')

    # Get terms per topic
    topics = lda_model.topicsMatrix()
    vocabArray = cvmodel.vocabulary

    wordNumbers = 15  # number of words per topic
    topicIndices = lda_model.describeTopics(maxTermsPerTopic=wordNumbers).rdd.map(tuple)

    topics_final = topicIndices.map(lambda topic: topic_render(topic, wordNumbers, vocabArray)).collect()

    for topic in range(len(topics_final)):
        print("Topic" + str(topic) + ":")
        print(topics_final[topic])

    print("Completed in {} min".format((time() - start) / 60))


if __name__ == '__main__':
    main()
