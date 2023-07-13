# link tham khao: https://phamdinhkhanh.github.io/2019/07/15/PySparkSQL.html
from pyspark import SparkContext
from pyspark.sql import SparkSession
# Stop spark if it existed.
try:
    sc.stop()
except:
    print('sc have not yet created!')

# set connection
sc = SparkContext(master = "local", appName = "First app")
# Check spark context
print(sc)
# Check spark context version
print(sc.version)
sc = SparkContext.getOrCreate()
print(sc)

# set session
my_spark = SparkSession.builder.getOrCreate()
# Print my_spark session
print(my_spark)

# List all table exist in spark sesion
# print(my_spark.catalog.listTables())

flights = my_spark.read.csv('flights.csv', header = True)
# show fligths top 5
# flights.show(5)

# list all table exist in spark session
print("luc nay chua co trong catalog cua cluster", my_spark.catalog.listTables())

# Create a temporary table on catalog of local data frame flights as new temporary table flights_temp on catalog
flights.createOrReplaceTempView('flights_temp')
# check list all table available on catalog
print("luc nay da tao tam thoi trong catalog cua cluster", my_spark.catalog.listTables())

avg_speed = flights.select("ORIGIN_AIRPORT", "DESTINATION_AIRPORT", "TAIL_NUMBER", (flights.DISTANCE).alias("avg_speed"))
avg_speed.printSchema()
# avg_speed.show(5)

# loc chuyen bay xuat phat tu SEA va diem den ANC
filter_SEA_ANC = flights.filter("ORIGIN_AIRPORT == 'SEA'") \
                        .filter("DESTINATION_AIRPORT == 'ANC'")
# filter_SEA_ANC.show(5)

# CAC LENH BIEN DOI CUA spark.sql()
flights_10 = my_spark.sql('SELECT * FROM flights_temp WHERE AIR_TIME > 10')
# flights_10.show(5)

# print(flights.printSchema())

print('Shape of previous data: ({}, {})'.format(flights.count(), len(flights.columns)))
flights_SEA = my_spark.sql("select ARRIVAL_DELAY, ARRIVAL_TIME, MONTH, YEAR, DAY_OF_WEEK, DESTINATION_AIRPORT, AIRLINE from flights_temp where ORIGIN_AIRPORT = 'SEA' and AIRLINE in ('DL', 'AA') ")
print('Shape of flights_SEA data: ({}, {})'.format(flights_SEA.count(), len(flights_SEA.columns)))

# Create boolean variable IS_DELAY variable as Target
flights_SEA = flights_SEA.withColumn("IS_DELAY", flights_SEA.ARRIVAL_DELAY > 0)
# Now Convert Boolean variable into integer
flights_SEA = flights_SEA.withColumn("label", flights_SEA.IS_DELAY.cast("integer"))
# Remove missing value
model_data = flights_SEA.filter("ARRIVAL_DELAY is not null \
                                and ARRIVAL_TIME is not null \
                                and MONTH is not null \
                                and YEAR is not null  \
                                and DAY_OF_WEEK is not null \
                                and DESTINATION_AIRPORT is not null \
                                and AIRLINE is not null")

print('Shape of model_data data: ({}, {})'.format(model_data.count(), len(model_data.columns)))

# ARRIVAL_TIME, MONTH, YEAR, DAY_OF_WEEK
model_data = model_data.withColumn("ARRIVAL_TIME", model_data.ARRIVAL_TIME.cast("integer"))
model_data = model_data.withColumn("MONTH", model_data.MONTH.cast("integer"))
model_data = model_data.withColumn("YEAR", model_data.YEAR.cast("integer"))
model_data = model_data.withColumn("DAY_OF_WEEK", model_data.DAY_OF_WEEK.cast("integer"))
model_data.printSchema()

from pyspark.ml.feature import StringIndexer, OneHotEncoder
# I. With DESTINATION_AIRPORT
# Create StringIndexer
dest_indexer = StringIndexer(inputCol = "DESTINATION_AIRPORT", \
                             outputCol = "DESTINATION_INDEX")

# Create OneHotEncoder
dest_onehot = OneHotEncoder(inputCol = "DESTINATION_INDEX", \
                            outputCol = "DESTINATION_FACT")

# II. With AIRLINE
# Create StringIndexer
airline_indexer = StringIndexer(inputCol = "AIRLINE", \
                                outputCol = "AIRLINE_INDEX")

# Create OneHotEncoder
airline_onehot = OneHotEncoder(inputCol = "AIRLINE_INDEX", \
                               outputCol = "AIRLINE_FACT")

# Make a VectorAssembler
from pyspark.ml.feature import VectorAssembler
vec_assembler = VectorAssembler(inputCols = ["ARRIVAL_TIME", "MONTH", "YEAR", \
                                             "DAY_OF_WEEK", "DESTINATION_FACT",\
                                             "AIRLINE_FACT"], 
                                outputCol = "features")

from pyspark.ml import Pipeline

# Make a pipeline
flights_sea_pipe  = Pipeline(stages = [dest_indexer, dest_onehot, airline_indexer, \
                                       airline_onehot, vec_assembler])

# Huấn luyện và đánh giá model.
# create pipe_data from pipeline
pipe_data = flights_sea_pipe.fit(model_data).transform(model_data)
# Split train/test data
train, test = pipe_data.randomSplit([0.8, 0.2])

from pyspark.ml.classification import LogisticRegression
# Create logistic regression
lr = LogisticRegression()

# Import the evaluation submodule
import pyspark.ml.evaluation as evals

# Create a BinaryClassificationEvaluator
evaluator = evals.BinaryClassificationEvaluator(metricName = "areaUnderROC")

# Tuning model thông qua param gridSearch.
# Import the tuning submodule
import pyspark.ml.tuning as tune
import numpy as np

# Create the parameter grid
grid = tune.ParamGridBuilder()

# Add the hyperparameter, we can add more than one hyperparameter
grid = grid.addGrid(lr.regParam, np.arange(0, 0.1, 0.01))

# Build the grid
grid = grid.build()

# Create the CrossValidator
cv = tune.CrossValidator(estimator=lr,
               estimatorParamMaps=grid,
               evaluator=evaluator
               )

# Fit cross validation on models
models = cv.fit(train)

best_lr = models.bestModel
best_lr = lr.fit(train)
print(best_lr)
# Use the model to predict the test set
test_results = best_lr.transform(test)

# Evaluate the predictions
print(evaluator.evaluate(test_results))
