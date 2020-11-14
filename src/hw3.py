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

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import glob
import json

from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.sql import functions as F
from pyspark.sql.types import StringType
from pyspark.ml.feature import CountVectorizer, Tokenizer, StopWordsRemover
from pyspark.sql.functions import monotonically_increasing_id
from pyspark.ml.clustering import LDA
from pyspark.ml.linalg import Vectors, SparseVector
from pyspark.sql.functions import col, size

# Spark setup
conf = SparkConf().setMaster("local").setAppName("HW3-data-exploration")
sc = SparkContext(conf=conf)
spark = SparkSession(sc)

# Get all json files
root_path = '../data/archive/'
all_json = glob.glob(f'{root_path}/**/*.json', recursive=True)
print("There are ", len(all_json), "sources files.")
#todo - for now restrict this to 100 files
# all_json = all_json[:100]

data = spark.read.json(all_json, multiLine=True)
data.createOrReplaceTempView("data")

# Select text columns
covid_sql = spark.sql(
        """
        SELECT
            metadata.title AS title,
            body_text.text AS body_text,
            paper_id
        FROM data
        """)

word_join_f = F.udf(lambda x: [''.join(w) for w in x], StringType())
covid_sql = covid_sql.withColumn("body_text", word_join_f("body_text"))

# todo - include word count graph here


# Tokenize the text in the text column
tokenizer = Tokenizer(inputCol="body_text", outputCol="words")
token_DataFrame = tokenizer.transform(covid_sql)

# Remove stopwords
remover = StopWordsRemover(inputCol="words", outputCol="filtered")
cleaned_DataFrame = remover.transform(token_DataFrame)

# Count vectorizer
cv_tmp = CountVectorizer(inputCol="filtered", outputCol="features")
cvmodel = cv_tmp.fit(cleaned_DataFrame)
df_vect = cvmodel.transform(cleaned_DataFrame)

# todo - need to add id column here?


# Fit the LDA Model
num_topics = 10
max_iterations = 50
lda = LDA(seed=1, optimizer="em", k=num_topics, maxIter=max_iterations)
lda_model = lda.fit(df_vect)

# Get terms per topic
topics = lda_model.topicsMatrix()
vocabArray = cvmodel.vocabulary

wordNumbers = 15  # number of words per topic
topicIndices = lda_model.describeTopics(maxTermsPerTopic = wordNumbers).rdd.map(tuple)


def topic_render(topic):  # specify vector id of words to actual words
    terms = topic[1]
    result = []
    for i in range(wordNumbers):
        term = vocabArray[terms[i]]
        result.append(term)
    return result

topics_final = topicIndices.map(lambda topic: topic_render(topic)).collect()

for topic in range(len(topics_final)):
    print ("Topic" + str(topic) + ":")
    for term in topics_final[topic]:
        print (term)
    print ('\n')
