# Databricks notebook source
import mlflow
import json

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

# MAGIC %md #### Install Huggingface repo locally so that MLflow pyfunc can load it

# COMMAND ----------

base_data_dir = "dbfs:/home/tim.lortz@databricks.com/huggingface/bert-base-NER/"

# COMMAND ----------

# MAGIC %sh 
# MAGIC git lfs install;
# MAGIC cd /tmp;
# MAGIC git clone https://huggingface.co/dslim/bert-base-NER;
# MAGIC ls -lth

# COMMAND ----------

# MAGIC %sh ls -lth /tmp/bert-base-NER

# COMMAND ----------

# MAGIC %fs ls file:/tmp/bert-base-NER

# COMMAND ----------

dbutils.fs.mkdirs(base_data_dir)
dbutils.fs.mv("file:/tmp/bert-base-NER",base_data_dir, recurse=True)

# COMMAND ----------

dbutils.fs.ls(base_data_dir)

# COMMAND ----------

# from pyfunc_modules.ner_transformers import TransformerNERModel

# COMMAND ----------

# MAGIC %md Read in conda env file as json, convert to dict

# COMMAND ----------

f = open("./conda_envs/conda_ner.json")
conda_env = json.load(f)
f.close()
conda_env

# COMMAND ----------

# loader_module_file = open("./pyfunc_modules/ner_transformers.py")
# for line in loader_module_file:
#   print(line)
# loader_module_file.close()

# COMMAND ----------

with mlflow.start_run(run_name="NER") as run:
  
  # reference class (file in repo?) as loader module
  
  mlflow.pyfunc.log_model(artifact_path="ner", loader_module="ner_transformers", conda_env=conda_env, code_path=["/Workspace/Repos/tim.lortz@databricks.com/transformers-mlflow-templates/pyfunc_modules/ner_transformers.py"], data_path=base_data_dir.replace("dbfs:", "/dbfs"))

# will loader module from repo work, or does it need to be in code path? can I dynamically build code path based on my user ID and repo name? 

# COMMAND ----------

run.info.run_id

# COMMAND ----------

ner_loaded = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/ner")

# COMMAND ----------

import pandas as pd
test_df = pd.DataFrame({"id":[1,2,3], "content": ["Abraham Lincoln cut down a cherry tree", "Florida has nice beaches", "Elon Musk owns Tesla"]})

results = ner_loaded.predict(test_df)

pd.set_option('display.max_colwidth', None)
results.head()
