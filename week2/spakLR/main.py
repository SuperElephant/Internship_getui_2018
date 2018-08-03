
"""
:author Chao Xiang
:date 2018-07

This file is used to train a Logistic Regression model for label base classification
CountVectorizer -> ChiSqSelector -> LR

1. split text in to a list of String (formating.py)
2. use CountVectorizer to convert text list in to a matrix of token counts (long binary vector)
3. use ChiSqSelector select features that matters
4. train Logistic Regression model

"""


from __future__ import division
from pyspark.sql import SparkSession
from formating import formating, mergeWithTag
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel \
    , BinaryLogisticRegressionSummary, LogisticRegression
from pyspark.sql.types import IntegerType
import matplotlib.pyplot as plt
from pyspark.sql import Row, SparkSession, functions, types
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml import Pipeline

# # -------without pipline-------------
# from formating import selector, count_vectorize


spark = SparkSession.builder.getOrCreate()

data0 = formating("/Users/chaoxiang/Desktop/sample_data/20180525_ml_data_0.txt", spark)
data1 = formating("/Users/chaoxiang/Desktop/sample_data/20180525_ml_data_1.txt", spark)

all_data = mergeWithTag(data0, data1, 0, 1)

# --------with pipline----------------
training_data, test_data = all_data.randomSplit([0.7, 0.3])

count_vectorizer = CountVectorizer(inputCol="tags", outputCol="features")
selector = ChiSqSelector(selectorType="percentile", percentile=0.5, featuresCol="features",
                         outputCol="selectedFeatures", labelCol="target")
lr = LogisticRegression(featuresCol="selectedFeatures", labelCol="target", regParam=0.1)

lr_pipline = Pipeline().setStages([count_vectorizer, selector, lr])
lr_pip_model = lr_pipline.fit(training_data)
lr_prediction = lr_pip_model.transform(test_data)

training_summary = lr_pip_model.stages[-1].summary

# # -------without pipline-------------
# all_data = count_vectorize(all_data)
# all_data.show(truncate = False)
#
# print all_data.count()
#
# all_data = selector(all_data, 0.5)
#
# training_data, test_data = all_data.randomSplit([0.7, 0.3])
#
# lr = LogisticRegression(featuresCol="selectedFeatures", labelCol="target", regParam=0.1)
# print "LR parameters:\n" + lr.explainParams()
#
# lr_model = lr.fit(training_data)
# lr_prediction = lr_model.transform(test_data)
#
# lr_prediction.show()
#
# training_summary = lr_model.summary

objective_history = training_summary.objectiveHistory
plt.plot(range(len(objective_history)), objective_history)
plt.show()
print "AreaUnderROC: "
print training_summary.areaUnderROC

df = lr_prediction.withColumn("prediction", lr_prediction["prediction"].cast(IntegerType()))
accuracy = df.filter("prediction = target").count() / df.count()

print "accuracy: "
print accuracy

# lr_model.save('/Users/chaoxiang/Desktop/sample_data/lr_model')


# new_df.createOrReplaceTempView("temp")
