# Databricks notebook source
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from mlflow.models import infer_signature
from databricks.feature_store import FeatureStoreClient

# COMMAND ----------

# 피쳐 스토어 데이터 로드
fs = FeatureStoreClient()
df = fs.read_table(name='edu_titanic.04_feature.fs_table')
df = df.toPandas()
df.head()

# COMMAND ----------

df.isnull().sum()

# COMMAND ----------

# Label Encoding
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Sex'] = le.fit_transform(df['Sex'])
df['Embarked'] = le.fit_transform(df['Embarked'])

# COMMAND ----------

X = df.drop(columns=['Survived', 'PassengerId', 'Sex', 'Embarked'])
y = df['Survived']
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=2024, shuffle=True)

# COMMAND ----------

run_name = 'edu_model_1'
with mlflow.start_run(run_name=run_name) as run:
    random_state = 2024
    dtc = LogisticRegression(random_state=random_state)
    dtc.fit(train_x, train_y)
    y_pred_class = dtc.predict(test_x)
    accuracy = accuracy_score(test_y, y_pred_class)
    f1 = f1_score(test_y, y_pred_class)

    print(f1)

    mlflow.log_param('random_state', random_state)
    mlflow.log_metric('accuracy', accuracy)
    mlflow.log_metric('f1_score', f1)
    
    signature = infer_signature(train_x, train_y)
    example = test_x[0:1]
    mlflow.sklearn.log_model(dtc, 'model', signature=signature, input_example=example)    


# COMMAND ----------

# MAGIC %md
# MAGIC Inference table 만들기

# COMMAND ----------

import mlflow

from mlflow.tracking.client import MlflowClient #클라이언트 트래킹
client = MlflowClient()

run_id = run.info.run_id
model_uri = f'runs:/{run_id}/model'
# 모델 로드
model_loaded = mlflow.pyfunc.load_model(model_uri=model_uri)

# COMMAND ----------

pred = model_loaded.predict(test_x)

# COMMAND ----------

inference_table = test_x.copy()
inference_table['predict'] = pred
inference_table['label'] = test_y
inference_table['model_id'] = '02_Model Logistic Regression'

import datetime
now = datetime.datetime.now()
formatted_now = now.strftime('%Y-%m-%d %H:%M:%S')
inference_table['timestamp'] = formatted_now

# COMMAND ----------

inference_table

# COMMAND ----------

# 카탈로그 저장
from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark_df = spark.createDataFrame(inference_table)
spark_df.write.format("delta").mode("append").saveAsTable('edu_titanic.05_results.infer_table')

# COMMAND ----------

# push pull test

# COMMAND ----------

# 모델선정 완료
