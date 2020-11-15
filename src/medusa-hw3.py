import os
from time import time
from pyspark.ml.clustering import LDA
from pyspark.ml.feature import CountVectorizer, Tokenizer, StopWordsRemover
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StringType


def init_spark():
    spark = SparkSession.builder.master('yarn').appName("HW3-data-exploration").getOrCreate()
    return spark


def read_json_files(root_path, spark):
    json_dir = root_path + "document_parses/pdf_json/"
    filenames = os.listdir(json_dir)

    all_json = [json_dir + filename for filename in filenames]
    print len(all_json)
    # todo - for now restrict this to 100 files
    all_json = all_json[:1000]

    data = spark.read.json(all_json, multiLine=True)
    data.createOrReplaceTempView("data")
    return get_title_abstract_text(spark, data)


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


# def topic_render(topic, wordNumbers, vocabArray):  # specify vector id of words to actual words
#     terms = topic[0]
#     print(terms)
#     result = []
#     for i in range(wordNumbers):
#         term = vocabArray[terms[i]]
#         result.append(term)
#     return result


def topic_render(topic, wordNumbers, vocabArray):  # specify vector id of words to actual words
    terms = topic[1]
    result = []
    for i in range(wordNumbers):
        term = vocabArray[terms[i]]
        result.append(term)
    return result


def main():
    start = time()

    root_path = './data/archive/'
    spark = init_spark()
    json_files = read_json_files(root_path, spark)
    data = get_title_abstract_text(spark, json_files)
    data = data.na.drop(subset=["body_text"])

    word_join_f = F.udf(lambda x: [''.join(w) for w in x], StringType())
    data = data.withColumn("body_text", word_join_f("body_text"))

    tokenizer = Tokenizer(inputCol="body_text", outputCol="words")
    token_DataFrame = tokenizer.transform(data)


    '''
    todo - do more to remove things here:
    '''
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

    # Get terms per topic
    topics = lda_model.topicsMatrix()
    vocabArray = cvmodel.vocabulary

    wordNumbers = 15  # number of words per topic
    topicIndices = lda_model.describeTopics(maxTermsPerTopic=wordNumbers).rdd.map(tuple)

    topics_final = topicIndices.map(lambda topic: topic_render(topic, wordNumbers, vocabArray)).collect()

    for topic in range(len(topics_final)):
        print "Topic" + str(topic) + ":"
        for term in topics_final[topic]:
            print(term)
        print('\n')

    print "Completed in {} min".format((time() - start) / 60)


if __name__ == '__main__':
    main()