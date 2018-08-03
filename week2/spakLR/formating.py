"""
:author Chao Xiang
:date 2018-07

Work with main.py
"""


from pyspark.sql import Row, SparkSession, functions, types
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import ChiSqSelector


def _convertToRow(line):
    """
    used to build a Row
    """
    rel = {}
    if len(line) == 5:
        rel['gid'] = line[0]
        rel['tags'] = line[1].split(',') + line[2].split(',') + line[3].split(',') + line[4].split(',')
    return rel


def formating(file_path, spark):
    """
    Read text file and generate DataFrame
    :param file_path: file path
    :return: DataFrame
    """
    raw_data = spark.sparkContext.textFile('file://' + file_path)
    data = raw_data.map(lambda line: line.split('|')).map(lambda line: Row(**_convertToRow(line))).toDF()
    data.first()
    return data
    # count_vectorizer = CountVectorizer(inputCol="tags", outputCol="featu3res")4
    # result = count_vectorizer.fit(data).transform(data)
    # return result


# # ----without pipline------------
# def count_vectorize(data):
#     count_vectorizer = CountVectorizer(inputCol="tags", outputCol="features")
#     result = count_vectorizer.fit(data).transform(data)
#     return result
#
#
# def selector(data, percentile=1):
#     selector = ChiSqSelector(selectorType="percentile", percentile=percentile, featuresCol="features",
#                              outputCol="selectedFeatures", labelCol="target").fit(data)
#     result = selector.transform(data)
#     return result


def mergeWithTag(df1, df2, tag1, tag2):
    """
    Merger two DataFrame with corresponding tags
    :param df1: First DF
    :param df2: Second DF
    :param tag1: Integer only
    :param tag2: Integer only
    :return: merged DF
    """
    print df1.count()
    print df2.count()
    tmp1_udf = functions.udf(lambda a: tag1, types.IntegerType())
    tmp2_udf = functions.udf(lambda a: tag2, types.IntegerType())
    df1 = df1.withColumn("target", tmp1_udf("gid"))
    df2 = df2.withColumn("target", tmp2_udf("gid"))
    return df1.union(df2)


if __name__ == '__main__':
    """
    test only for --without pipline--
    """
    spark = SparkSession.builder.getOrCreate()
    data0 = formating("/Users/chaoxiang/Desktop/sample_data/20180525_ml_data_0.txt", spark)
    data1 = formating("/Users/chaoxiang/Desktop/sample_data/20180525_ml_data_1.txt", spark)
    new_df = mergeWithTag(data0, data1, 0, 1)
    new_df = count_vectorize(new_df)
    new_df.show(truncate=False)
    print new_df.count()
    # new_df = selector(new_df)
    new_df.createOrReplaceTempView("temp")
    a = spark.sql("select * from temp where target = '1'")
    a.show()
    a = spark.sql("select * from temp where target = '0'")
    a.show()
