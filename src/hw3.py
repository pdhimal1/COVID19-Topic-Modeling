"""
Data from

COVID-19 Open Research Dataset Challenge (CORD-19)
https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge

Prakash Dhimal
Manav Garkel
George Mason University
CS 657 Mining Massive Datasets
Assignment 3: Topic Modeling
"""
import os
from time import time

import numpy as np
import pyLDAvis
from nltk import PorterStemmer
from nltk.corpus import stopwords
from pyspark import SparkContext
from pyspark.ml.clustering import LDA
from pyspark.ml.feature import CountVectorizer, Tokenizer, StopWordsRemover, IDF
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import explode, size
from pyspark.sql.types import StringType

stop_words = set(stopwords.words('english'))


def init_spark():
    SparkContext.setSystemProperty('spark.local.dir', '/home/dhimal/spark')
    spark = SparkSession.builder \
        .master("local[*]") \
        .config("spark.executor.memory", "32g") \
        .config("spark.driver.memory", "32g") \
        .config("spark.memory.offHeap.enabled", True) \
        .config("spark.memory.offHeap.size", "32g") \
        .appName("hw3") \
        .getOrCreate()
    return spark


def read_json_files(root_path, spark):
    json_dir = root_path + "document_parses/pdf_json/"
    filenames = os.listdir(json_dir)

    all_json = [json_dir + filename for filename in filenames]
    # todo - for now restrict this to 100 files
    all_json = all_json[:1000]

    data = spark.read.json(all_json, multiLine=True)
    data.createOrReplaceTempView("data")
    return data


def get_body_text(spark, data):
    body_text_only_data = spark.sql(
        """
            SELECT
                body_text.text AS body_text,
                paper_id
            FROM data
            """)
    return body_text_only_data


def topic_render(topic, wordNumbers, vocabArray):  # specify vector id of words to actual words
    terms = topic[1]
    result = []
    for i in range(wordNumbers):
        term = vocabArray[terms[i]]
        result.append(term)
    return result


def clean_up_sentences(sentence):
    stemmer = PorterStemmer()
    matches = [word for word in sentence.split(' ') if word.isalnum()]
    matches = [word.lower() for word in matches]
    matches = [word for word in matches if word not in stop_words]
    matches = [stemmer.stem(word) for word in matches]
    matches = [word for word in matches if len(word) >= 4]
    return matches


def format_data_to_pyldavis(cleaned_DataFrame, cvmodel, lda_transformed, lda_model):
    counts = cleaned_DataFrame.select((explode(cleaned_DataFrame.filtered)).alias("tokens")).groupby("tokens").count()
    wc = {i['tokens']: i['count'] for i in counts.collect()}
    wc = [wc[x] for x in cvmodel.vocabulary]

    data = {'topic_term_dists': np.array(lda_model.topicsMatrix().toArray()).T,
            'doc_topic_dists': np.array(
                [x.toArray() for x in lda_transformed.select(["topicDistribution"]).toPandas()['topicDistribution']]),
            'doc_lengths': [x[0] for x in cleaned_DataFrame.select(size(cleaned_DataFrame.filtered)).collect()],
            'vocab': cvmodel.vocabulary,
            'term_frequency': wc}

    return data


def clean_up(document):
    cleaned = [clean_up_sentences(w) for w in document]
    joined = [' '.join(w) for w in cleaned]
    return joined


def main():
    timeStamp = str(int(time()))
    out_file_name = '../out/output-' + timeStamp + '.txt'
    out_file = open(out_file_name, 'w')

    start = time()
    root_path = '../data/archive/'
    spark = init_spark()
    json_files = read_json_files(root_path, spark)
    data = get_body_text(spark, json_files)
    print("data reading done")

    # clean the data
    word_clean_up_F = F.udf(lambda x: clean_up(x), StringType())
    data = data.withColumn("body_text_cleaned", word_clean_up_F("body_text"))
    print("data processing done")

    tokenizer = Tokenizer(inputCol="body_text_cleaned", outputCol="words")
    token_DataFrame = tokenizer.transform(data)

    # Remove stopwords
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    cleaned_DataFrame = remover.transform(token_DataFrame)

    # Count vectorizer
    cv_tmp = CountVectorizer(inputCol="filtered", outputCol="count_features")
    cvmodel = cv_tmp.fit(cleaned_DataFrame)
    count_dataframe = cvmodel.transform(cleaned_DataFrame)

    # TF-IDF Vectorizer
    tfidf = IDF(inputCol="count_features", outputCol="features")
    tfidfmodel = tfidf.fit(count_dataframe)
    tfidf_dataframe = tfidfmodel.transform(count_dataframe)

    print("Ready to fit with the LDA model")
    # Fit the LDA Model
    num_topics = 10
    max_iterations = 10
    lda_start = time()
    lda = LDA(seed=1, optimizer="em", k=num_topics, maxIter=max_iterations)
    lda_model = lda.fit(tfidf_dataframe)
    lda_transformed = lda_model.transform(tfidf_dataframe)
    lda_end = time()
    print("LDA complete")
    # joblib.dump(lda_model, 'lda.csv')

    # Get terms per topic
    topics = lda_model.topicsMatrix()
    vocabArray = cvmodel.vocabulary

    wordNumbers = 15  # number of words per topic
    topicIndices = lda_model.describeTopics(maxTermsPerTopic=wordNumbers).rdd.map(tuple)

    topics_final = topicIndices.map(lambda topic: topic_render(topic, wordNumbers, vocabArray)).collect()

    for topic in range(len(topics_final)):
        print("Topic " + str(topic) + ":")
        print("Topic " + str(topic) + ":", file=out_file)
        print(topics_final[topic])
        print(topics_final[topic], file=out_file)

    print("Full runtime : {} min. ".format((time() - start) / 60))
    print("LDA runtime : {} min. ".format((lda_end - lda_start) / 60))
    print("Check" + out_file.name)

    # Data Visualization
    data = format_data_to_pyldavis(cleaned_DataFrame, cvmodel, lda_transformed, lda_model)
    py_lda_prepared_data = pyLDAvis.prepare(**data)
    file_name = '../out/data-viz-' + timeStamp + '.html'
    pyLDAvis.save_html(py_lda_prepared_data, file_name)
    pyLDAvis.show(py_lda_prepared_data)


if __name__ == '__main__':
    main()
