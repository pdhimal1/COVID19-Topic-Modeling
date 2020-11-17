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
from nltk.corpus import stopwords
from pyspark.ml.clustering import LDA
from pyspark.ml.feature import CountVectorizer, Tokenizer, StopWordsRemover, IDF
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import explode, size
from pyspark.sql.types import StringType

stop_words = set(stopwords.words('english'))


def init_spark():
    # SparkContext.setSystemProperty('spark.local.dir', '<>')
    spark = SparkSession.builder \
        .master("local") \
        .config("spark.executor.memory", "16g") \
        .config("spark.driver.memory", "16g") \
        .appName("hw3") \
        .getOrCreate()
    return spark


def read_json_files(root_path, spark, num):
    json_dir = root_path + "document_parses/pdf_json/"
    filenames = os.listdir(json_dir)

    all_json = [json_dir + filename for filename in filenames]
    all_json = all_json[:num]

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
    matches = [word for word in sentence.split(' ') if word.isalnum()]
    matches = [word.lower() for word in matches]
    matches = [word for word in matches if word not in stop_words]
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


# See https://stackoverflow.com/questions/41819761/pyldavis-visualization-of-pyspark-generated-lda-model
def filter_bad_docs(data):
    bad = 0
    doc_topic_dists_filtrado = []
    doc_lengths_filtrado = []

    for x, y in zip(data['doc_topic_dists'], data['doc_lengths']):
        if np.sum(x) == 0:
            bad += 1
        elif np.sum(x) != 1:
            bad += 1
        elif np.isnan(x).any():
            bad += 1
        else:
            doc_topic_dists_filtrado.append(x)
            doc_lengths_filtrado.append(y)

    data['doc_topic_dists'] = doc_topic_dists_filtrado
    data['doc_lengths'] = doc_lengths_filtrado


def clean_up(document):
    cleaned = [clean_up_sentences(w) for w in document]
    joined = [' '.join(w) for w in cleaned]
    return joined


def main():
    timeStamp = str(int(time()))
    # todo
    num = 100
    out_file_name = '../out/output-' + timeStamp + "-" + str(num) + '.txt'
    out_file = open(out_file_name, 'w')

    start = time()
    root_path = '../data/archive/'
    spark = init_spark()
    json_files = read_json_files(root_path, spark, num)
    data = get_body_text(spark, json_files)
    print("data reading done")

    # clean the data
    word_clean_up_F = F.udf(lambda x: clean_up(x), StringType())
    data = data.withColumn("body_text_cleaned", word_clean_up_F("body_text"))
    data = data.select("body_text_cleaned")
    print("data processing done")

    tokenizer = Tokenizer(inputCol="body_text_cleaned", outputCol="words")
    token_DataFrame = tokenizer.transform(data)
    token_DataFrame = token_DataFrame.select("words")

    # Remove stopwords
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    cleaned_DataFrame = remover.transform(token_DataFrame)
    cleaned_DataFrame = cleaned_DataFrame.select('filtered')

    # Count vectorizer
    cv_tmp = CountVectorizer(inputCol="filtered", outputCol="count_features")
    cvmodel = cv_tmp.fit(cleaned_DataFrame)
    count_dataframe = cvmodel.transform(cleaned_DataFrame)
    count_dataframe = count_dataframe.select('count_features')

    # TF-IDF Vectorizer
    tfidf = IDF(inputCol="count_features", outputCol="features")
    tfidfmodel = tfidf.fit(count_dataframe)
    tfidf_dataframe = tfidfmodel.transform(count_dataframe).select("features")

    print("Ready to fit with the LDA model")
    # Fit the LDA Model
    num_topics = 5
    max_iterations = 20
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

    # cleaned_DataFrame.write.csv('cleaned_DataFrame' + timeStamp + "-" + str(num) + '.csv')
    # cvmodel.save('cvmodel' + timeStamp + "-" + str(num) + '.csv')
    # lda_transformed.write.csv('lda_transformed' + timeStamp + "-" + str(num) + '.csv')
    # lda_model.write.csv('lda_model' + timeStamp + "-" + str(num) + '.csv')
    cleaned_DataFrame.cache()
    lda_transformed.cache()

    # Data Visualization
    data = format_data_to_pyldavis(cleaned_DataFrame, cvmodel, lda_transformed, lda_model)
    print("Preparing data with pyLDAvis ...")
    filter_bad_docs(data)
    py_lda_prepared_data = pyLDAvis.prepare(**data)
    file_name = '../out/data-viz-' + timeStamp + '.html'
    print("Saving pyLDAvis html page ...")
    pyLDAvis.save_html(py_lda_prepared_data, file_name)
    pyLDAvis.show(py_lda_prepared_data)
    spark.stop()


if __name__ == '__main__':
    main()
