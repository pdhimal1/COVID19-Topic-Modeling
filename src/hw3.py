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

import matplotlib.pyplot as plt

plt.style.use('ggplot')
import glob

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
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
    covid_sql = spark.sql(
        """
            SELECT
                metadata.title AS title,
                abstract.text AS abstract, 
                body_text.text AS body_text,
                back_matter.text AS back_matter,
                paper_id
            FROM data
            """)
    return covid_sql


# Adding the Word Count Column
def add_word_count(data):
    word_join_f = F.udf(lambda x: [''.join(w) for w in x], StringType())

    data_df = data.withColumn("abstract", word_join_f("abstract"))
    data_df = data_df.withColumn("body_text", word_join_f("body_text"))

    # see https://stackoverflow.com/questions/48927271/count-number-of-words-in-a-spark-dataframe
    data_df = data_df.withColumn('wordCount_abstract', F.size(F.split(F.col('abstract'), ' ')))
    data_df = data_df.withColumn('wordCount_body_text', F.size(F.split(F.col('body_text'), ' ')))

    return data_df


def main():
    root_path = '../data/archive/'
    spark = init_spark()
    meta_data = read_metadata(root_path, spark)
    json_files = read_json_files(root_path, spark)
    data = get_title_abstract_text(spark, json_files)
    data_wc = add_word_count(data)
    data_wc.show()
    data_wc.limit(100).toPandas()[['wordCount_abstract', 'wordCount_body_text']] \
        .plot(kind='box',
              title='Boxplot of Word Count',
              figsize=(10, 6))
    plt.show()


if __name__ == '__main__':
    main()
