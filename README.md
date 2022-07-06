# transformers-mlflow-templates

## Motivation
The rich ecosystem of Transformer models, e.g. those in Huggingface's [model hub](https://huggingface.co/models), make it increasingly possible to apply transfer learning to text and other unstructured data types. This is easy to do for simple input examples, such as a string, list or Pandas DataFrame of documents. However, real-life data pipelines are rarely so simple. The model inferencing step is nearly always preceded and succeeded by custom data transformation logic. Thus, the "model" being deployed needs to couple the pretrained transformer model with this custom transformation logic.

[MLflow](https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html) has a `python_function` model flavor that enables this exact use case - coupling model artifacts with custom data handling code into a single model that can be registered, versioned, shared, access-controlled, and exposed with a common set of APIs (e.g. a `predict()` method) regardless of the underlying libraries or logic being used. This project attempts to simplify the creation of Transformer-based MLflow models by providing templates for common use cases.

## Template structure
The templates are broken into the two basic components required by MLflow python functions: the conda environment specification, and the Python module called a loader module. Per the MLflow docs:
> The loader module defines a _load_pyfunc() method that performs the following tasks:
>
> - Load data from the specified data_path. For example, this process may include deserializing pickled Python objects or models or parsing CSV files.
>
> - Construct and return a pyfunc-compatible model wrapper. As in the first use case, this wrapper must define a predict() method that is used to evaluate queries. predict() must adhere to the Inference API.
>
> The loader_module parameter specifies the name of your loader module.

In the project, these are located under `/conda_envs` and `/pyfunc_modules`, respectively. In general, the conda environment specifications lend themselves well to re-use across templates (especially if using e.g. the [Transformers](https://pypi.org/project/transformers/) library), whereas the loader modules may have multiple implementations for a single pretrained Transformer model, based on the custom data handling logic.

## Example usage
### Creating a new template
The loader modules under `/loader_modules` and conda environment specifications under `/conda_envs` provide examples of new templates. 

### Registering a model from an existing template
Let's walk through a named entity recognition example, using this popular Huggingface Hub [model](https://huggingface.co/dslim/bert-base-NER/tree/main). The example builds on `/pyfunc_modules/ner_transformers.py`. This explanation will assume we're working in Databricks, using the DBFS file system and the Repos feature for simplicity with using MLflow. However, the same basic approach would work in a local development environment, once MLflow has been configured.

1. Download the Transformer model artifacts from their web source, e.g. GitHub or the Huggingface Hub. Note that Huggingface Hub leverages Git LFS to avoid storing copies of large pretrained models, so the appropriate model artifact (e.g. `pytorch_model.bin` for PyTorch or `tf_model.h5` for TensorFlow) will need to be downloaded manually. 
2. Place the directory of downloaded artifacts into an appropriate location, where MLflow can access them. For example, the NER model might go in `dbfs:/tmp/huggingface/bert-base-NER/`
3. Read in the appropriate conda environment specification json as a Python dictionary. Call this location `base_data_dir`

```
f = open("./conda_envs/conda_ner.json")
conda_env = json.load(f)
f.close()
```

4. Kick off an MLflow experiment run to log the model to the MLflow tracking server with all of its dependencies:

```
with mlflow.start_run(run_name="NER") as run:
  
    mlflow.pyfunc.log_model(artifact_path="ner", loader_module="ner_transformers", conda_env=conda_env, code_path=["/Workspace/Repos/<databricks username>/transformers-mlflow-templates/pyfunc_modules/ner_transformers.py"], data_path=base_data_dir.replace("dbfs:", "/dbfs"))
```

5. Load the logged model back from the experiment tracker: `ner_loaded = mlflow.pyfunc.load_model(f"runs:/{run.info.run_id}/ner")`
6. Apply the loaded model to new data. For example:

```
import pandas as pd
test_df = pd.DataFrame({"id":[1,2,3], "content": ["Abraham Lincoln cut down a cherry tree", "Florida has nice beaches", "Elon Musk owns Tesla"]})

results = ner_loaded.predict(test_df)

pd.set_option('display.max_colwidth', None)
results.head()
```