# Databricks notebook source
# MAGIC %md
# MAGIC ### Youtube comments analysis.
# MAGIC 
# MAGIC In this notebook, we have a dataset of user comments for youtube videos related to animals or pets. We will attempt to identify cat or dog owners based on these comments, find out the topics important to them, and then identify video creators with the most viewers that are cat or dog owners.

# COMMAND ----------

# MAGIC %md
# MAGIC The dataset provided for this coding test are comments for videos related to animals and/or pets. The dataset is 240MB compressed; please download the file using this google drive link:
# MAGIC https://drive.google.com/file/d/1o3DsS3jN_t2Mw3TsV0i7ySRmh9kyYi1a/view?usp=sharing
# MAGIC 
# MAGIC  The dataset file is comma separated, with a header line defining the field names, listed here:
# MAGIC ● creator_name. Name of the YouTube channel creator.
# MAGIC ● userid. Integer identifier for the users commenting on the YouTube channels.
# MAGIC ● comment. Text of the comments made by the users.
# MAGIC 
# MAGIC 
# MAGIC Step 1: Identify Cat And Dog Owners
# MAGIC Find the users who are cat and/or dog owners.
# MAGIC 
# MAGIC Step 2: Build And Evaluate Classifiers
# MAGIC Build classifiers for the cat and dog owners and measure the performance of the classifiers.
# MAGIC 
# MAGIC Step 3: Classify All The Users
# MAGIC Apply the cat/dog classifiers to all the users in the dataset. Estimate the fraction of all users
# MAGIC who are cat/dog owners.
# MAGIC 
# MAGIC Step 4: Extract Insights About Cat And Dog Owners
# MAGIC Find topics important to cat and dog owners.
# MAGIC 
# MAGIC Step 5: Identify Creators With Cat And Dog Owners In The Audience
# MAGIC Find creators with the most cat and/or dog owners. Find creators with the highest statistically
# MAGIC significant percentages of cat and/or dog owners.

# COMMAND ----------

# MAGIC %md
# MAGIC #### 0. Data Exploration and Cleaning

# COMMAND ----------

# read data
df = spark.read.load("/FileStore/tables/animals_comments_csv-5aaff.gz", format='csv', header = True, inferSchema = True)
df.show(10)

# COMMAND ----------

# check datatype
df.dtypes

# COMMAND ----------

# check data counts
df.count() 

# COMMAND ----------

# Count null values in each columns 
print('Number of null values in creator_name: ',df.filter(df['creator_name'].isNull()).count())
print('Number of null values in userid: ',df.filter(df['userid'].isNull()).count())
print('Number of null values in comment: ',df.filter(df['comment'].isNull()).count())

# COMMAND ----------

# drop out rows with no comments and no userid
def pre_process(df):
  df_drop = df.filter(df['comment'].isNotNull())
  df_drop = df_drop.filter(df_drop['userid'].isNotNull())
  df_drop = df_drop.dropDuplicates()
  
  print('After dropping, we have ', str(df_drop.count()), 'row in dataframe')
  return df_drop

df_drop = pre_process(df)

# COMMAND ----------

import pyspark.sql.functions as F
#convert text in comment to lower case.
df_clean = df_drop.withColumn('comment', F.lower(F.col('comment')))

# COMMAND ----------

display(df_clean)

# COMMAND ----------

# MAGIC %md
# MAGIC #####  This is an unlabeled dataset so we need to label each comment in advance since we want to train a classifier to identify cat and dog onwers
# MAGIC 1. label comment when he/she has dogs or cats.
# MAGIC 2. label comment when he/she don't have a dog or cat.
# MAGIC 3. Combine 1 and 2 as our training dataset, and rest of the dataset will be the data we predict.
# MAGIC 4. The strategy to tell if a user own or not own is just using key words (like I have a dog) to tell. Otherwise we can't have better ways and don't have labels.

# COMMAND ----------

# DBTITLE 1,Label the data
# find user with preference of dog and cat
# note: please propose your own approach and rule to label data 
cond = (df_clean["comment"].like("%my dog%") | df_clean["comment"].like("%i have a dog%")\
        | df_clean["comment"].like("%my cat%") | df_clean["comment"].like("%i have a cat%") \
        | df_clean["comment"].like("%my dogs%") | df_clean["comment"].like("%my cats%")\
        | df_clean["comment"].like("%my cat%") | df_clean["comment"].like("%i have dogs%")\
        | df_clean["comment"].like("%i have cats%") | df_clean["comment"].like("%my puppy%")\
        | df_clean["comment"].like("%my kitten%") | df_clean["comment"].like("%i have a puppy%")\
        | df_clean["comment"].like("%i have puppies%"))

df_clean = df_clean.withColumn('dog_cat',  cond)

# find user do not have 
df_clean = df_clean.withColumn('no_pet', ~df_clean["comment"].like("%my%") & ~df_clean["comment"].like("%have%") & ~df_clean["comment"].like("%my dog%") \
                              & ~df_clean["comment"].like("%my cat%")) 

# COMMAND ----------

df_clean.show()

# COMMAND ----------

# MAGIC %md
# MAGIC #### 1. Data preprocessing and Build the classifier 
# MAGIC To train a model against comments, we use RegexTokenizer to split each comment into a list of words and then use Word2Vec to convert the list to a word vector. Word2Vec map each word to a unique fixed-size vector and then transform each document into a vector using the average of all words in the document.

# COMMAND ----------

# data preprocessing 
from pyspark.ml.feature import RegexTokenizer

regexTokenizer = RegexTokenizer(inputCol="comment", outputCol="text", pattern="\\W")
df_clean = regexTokenizer.transform(df_clean)
df_clean.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Alert: First try is to use 1,000,000 rows for testing

# COMMAND ----------

from pyspark.sql.functions import rand 

df_clean.orderBy(rand(seed=0)).createOrReplaceTempView("table1")
df_clean = spark.sql("select * from table1 limit 1000000")

# COMMAND ----------

# use word2vec get text vector feature.
from pyspark.ml.feature import Word2Vec
# Learn a mapping from words to Vectors. (choose higher vectorSize here)
#word2Vec = Word2Vec(vectorSize=20, minCount=1, inputCol="text", outputCol="wordVector")
word2Vec = Word2Vec(vectorSize=50, minCount=1, inputCol="text", outputCol="wordVector")
model = word2Vec.fit(df_clean)

df_model = model.transform(df_clean)
df_model.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Get training dataset
# MAGIC We are using the training dataset as 'Has cat or dogs' + 'Do not have pets' which is 'dot_cat' is True + 'No_pets' is True
# MAGIC 
# MAGIC Rest of data will prepare for predicting

# COMMAND ----------

df_pets = df_model.filter(F.col('dog_cat') == True) 
df_no_pets = df_model.filter(F.col('no_pet') ==  True)
print("Number of confirmed user who own dogs or cats: ", df_pets.count())
print("Number of confirmed user who don't have pet's: ", df_no_pets.count())

# COMMAND ----------

df_pets.show()

# COMMAND ----------

df_no_pets.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC the number of negative labels is around 100 times more than positive labels, we need to downsampling the negative labels. The gap should not be more than 10 times. Here I set the balance ratio around 1:2 (1 for positive, 2 for negative)

# COMMAND ----------

from pyspark.sql.functions import rand 
df_no_pets.orderBy(rand()).createOrReplaceTempView("table")

Num_Pos_Label = df_model.filter(F.col('dog_cat') == True).count() 
Num_Neg_Label = df_model.filter(F.col('no_pet') ==  True).count()

#Q1 = spark.sql("SELECT col1 from table where col2>500 limit {}, 1".format(q25))
#pass variable to sql
df_no_pets_down = spark.sql("select * from table where limit {}".format(Num_Pos_Label*2))

# COMMAND ----------

print('Now after balancing the lables, we have ')   
print('Positive label: ', Num_Pos_Label)
print('Negtive label: ', df_no_pets_down.count())

# COMMAND ----------

def get_label(df_pets,df_no_pets_down):
  df_labeled = df_pets.select('dog_cat','wordVector').union(df_no_pets_down.select('dog_cat','wordVector'))
  return df_labeled

df_labeled = get_label(df_pets,df_no_pets_down)
df_labeled.show(10)

# COMMAND ----------

#convert Boolean value to 1 and 0's
from pyspark.sql.functions import col, udf
from pyspark.sql.types import IntegerType

def multiple(x):
  return int(x*1)
udf_boolToInt= udf(lambda z: multiple(z),IntegerType())
df_labeled = df_labeled.withColumn('label',udf_boolToInt('dog_cat'))
df_labeled.show(10)

# COMMAND ----------

# DBTITLE 1,Build ML model
# MAGIC %md
# MAGIC #### LogisticRegression

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit

train, test = df_labeled.randomSplit([0.8, 0.2], seed=12345)

lr = LogisticRegression(featuresCol="wordVector",labelCol="label" , maxIter=10, regParam=0.1, elasticNetParam=0.8)

# Run TrainValidationSplit, and choose the best set of parameters.
lrModel = lr.fit(train)

# Make predictions on test data. model is the model with combination of parameters
# that performed best.
predictions = lrModel.transform(test)
predictions.show(10)

# COMMAND ----------

# Extract the summary from the returned LogisticRegressionModel instance trained
# in the earlier example
trainingSummary = lrModel.summary

# # Obtain the objective per iteration
# objectiveHistory = trainingSummary.objectiveHistory
# print("objectiveHistory:")
# for objective in objectiveHistory:
#     print(objective)

# Obtain the receiver-operating characteristic as a dataframe and areaUnderROC.
trainingSummary.roc.show()
# print("areaUnderROC: " + str(trainingSummary.areaUnderROC))

# # Set the model threshold to maximize F-Measure
# fMeasure = trainingSummary.fMeasureByThreshold
# maxFMeasure = fMeasure.groupBy().max('F-Measure').select('max(F-Measure)').head()
# bestThreshold = fMeasure.where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure)']) \
#     .select('threshold').head()['threshold']
# lr.setThreshold(bestThreshold)

# COMMAND ----------

print("areaUnderROC: " + str(trainingSummary.areaUnderROC))

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator


def get_evaluation_result(predictions):
  evaluator = BinaryClassificationEvaluator(
      labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
  AUC = evaluator.evaluate(predictions)

  TP = predictions[(predictions["label"] == 1) & (predictions["prediction"] == 1.0)].count()
  FP = predictions[(predictions["label"] == 0) & (predictions["prediction"] == 1.0)].count()
  TN = predictions[(predictions["label"] == 0) & (predictions["prediction"] == 0.0)].count()
  FN = predictions[(predictions["label"] == 1) & (predictions["prediction"] == 0.0)].count()

  accuracy = (TP + TN)*1.0 / (TP + FP + TN + FN)
  precision = TP*1.0 / (TP + FP)
  recall = TP*1.0 / (TP + FN)


  print ("True Positives:", TP)
  print ("False Positives:", FP)
  print ("True Negatives:", TN)
  print ("False Negatives:", FN)
  print ("Test Accuracy:", accuracy)
  print ("Test Precision:", precision)
  print ("Test Recall:", recall)
  print ("Test AUC of ROC:", AUC)

print("Prediction result summary for Logistic Regression Model:  ")
get_evaluation_result(predictions)

# COMMAND ----------

# DBTITLE 0,Try random forest model
# MAGIC %md
# MAGIC #### RandomForest

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="label", featuresCol="wordVector", numTrees=15)

# Train model.  This also runs the indexers.
model = rf.fit(train)

# Make predictions.
predictions = model.transform(test)

# Select example rows to display.
predictions.show(10)

# COMMAND ----------

print("Prediction result summary for Random Forest Model:  ")
get_evaluation_result(predictions)

# COMMAND ----------

# DBTITLE 1,Apply model
# MAGIC %md
# MAGIC #### 2. Classify All The Users
# MAGIC Now applying model about cat/dog on other users in the dataset

# COMMAND ----------

# get dataset for prediction (note to exclude people we already know the label)
# Users we don't know yet are those who don't own dog&cat and no_pets attribute is also flase
df_unknow = df_model.filter((F.col('dog_cat') == False) & (F.col('no_pet') == False)) 
df_unknow = df_unknow.withColumn('label',df_unknow.dog_cat.cast('integer'))
print("There are {} users whose attribute is unclear.".format(df_unknow.count()))
pred_all = model.transform(df_unknow)
pred_all.show(10)

# COMMAND ----------

# MAGIC %md
# MAGIC Fraction of the users who are cat/dog owners (ML estimate):
# MAGIC 
# MAGIC Using
# MAGIC 
# MAGIC Num of owner labeled + Num of owner predicted / Total users in our used dataset 

# COMMAND ----------

#df.select("columnname").distinct().show()

#number of total user
total_user = df_model.select('userid').distinct().count()
#number of labeled owner
owner_labeled = df_pets.select('userid').distinct().count() 
#number of owner predicted
owner_pred = pred_all.filter(F.col('prediction') == 1.0).count()

fraction = (owner_labeled+owner_pred)/total_user
print('Fraction of the users who are cat/dog owners (ML estimate): ', round(fraction,3))

# COMMAND ----------

# DBTITLE 0,Look at the reasons from the text 
# MAGIC %md
# MAGIC #### 3. Get insigts of Users
# MAGIC Get all onwers (labeled as one as well as predicted as one) and get the words frequency

# COMMAND ----------

from pyspark.ml.feature import StopWordsRemover

df_all_owner = df_pets.select('text').union(pred_all.filter(F.col('prediction') == 1.0).select('text'))

stopwords_custom = ['im', 'get', 'got', 'one', 'hes', 'shes', 'dog', 'dogs', 'cats', 'cat', 'kitty', 'much', 'really', 'love','like','dont','know','want','thin',\
                    'see','also','never','go','ive']

remover1 = StopWordsRemover(inputCol="raw", outputCol="filtered")
core = remover1.getStopWords()
core = core + stopwords_custom
remover = StopWordsRemover(inputCol="text", outputCol="filtered",stopWords=core)
df_all_owner = remover.transform(df_all_owner)

wc = df_all_owner.select('filtered').rdd.flatMap(lambda a: a.filtered).countByValue()

# COMMAND ----------

df_all_owner.show(1)

# COMMAND ----------

# MAGIC %md
# MAGIC ##### Topic insight:
# MAGIC 1. Check the most frequent words in cat and dog owners
# MAGIC 2. Check wordcloud

# COMMAND ----------

#wcSorted = wc.sort(lambda a: a[1])
wcSorted = sorted(wc.items(), key=lambda kv: kv[1],reverse = True)
wcSorted

# COMMAND ----------

from wordcloud import WordCloud
import matplotlib.pyplot as plt

text = " ".join([(k + " ")*v for k,v in wc.items()])

wcloud = WordCloud(background_color="white", max_words=20000, collocations = False,
               contour_width=3, contour_color='steelblue',max_font_size=40)

# Generate a word cloud image
wcloud.generate(text)

# Display the generated image:
# the matplotlib way:
fig,ax0=plt.subplots(nrows=1,figsize=(12,8))
ax0.imshow(wcloud,interpolation='bilinear')

ax0.axis("off")
display(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC #### 4. Identify Creators With Cat And Dog Owners In The Audience

# COMMAND ----------

#Get all creators whenever the users label is True(cat/dog owner)
df_create = df_pets.select('creator_name').union(pred_all.filter(F.col('prediction') == 1.0).select('creator_name'))

df_create.createOrReplaceTempView("create_table")

#get count
create_count = spark.sql("select distinct creator_name, count(*) as Number\
                          from create_table \
                          group by creator_name \
                          order by Number DESC")

# COMMAND ----------

create_count.show()
