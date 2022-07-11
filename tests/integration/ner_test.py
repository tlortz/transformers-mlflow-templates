from logging import shutdown
import mlflow
import json
import pandas as pd
import shutil

base_data_dir = "/Users/tim.lortz/Documents/Technical/data/huggingface/bert-base-NER/"

f = open("../../conda_envs/conda_ner.json")
conda_env = json.load(f)
f.close()

def test_conda_env():
    assert isinstance(conda_env, dict)

with mlflow.start_run(run_name="NER") as run:
    mlflow.pyfunc.log_model(artifact_path="ner", 
        loader_module="ner_transformers", 
        conda_env=conda_env, code_path=["../../pyfunc_modules/ner_transformers.py"], 
        data_path=base_data_dir.replace("dbfs:", "/dbfs")
    )

ner_loaded = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/ner")

def test_model_load():
    assert(isinstance(ner_loaded, mlflow.pyfunc.PyFuncModel))

test_df = pd.DataFrame({"id":[1,2,3], "content": ["Abraham Lincoln cut down a cherry tree", "Florida has nice beaches", "Elon Musk owns Tesla"]})

results = ner_loaded.predict(test_df)

def test_model_inference():
    assert(results.columns.values.tolist() == ['id', 'content', 'ner_extracted'] and results.shape[0]==3)

# pd.set_option('display.max_colwidth', None)
# print(results.head())

# clean up
mlflow.delete_run(run.info.run_id)
# used in local environment tests. Not necessary when running in Databricks
shutil.rmtree(f"mlruns/0/{run.info.run_id}/", ignore_errors=True)