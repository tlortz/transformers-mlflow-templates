# transformers-mlflow-templates

## Motivation
The rich ecosystem of Transformer models, e.g. those in Huggingface's [model hub](https://huggingface.co/models), make it increasingly possible to apply transfer learning to text and other unstructured data types. This is easy to do for simple input examples, such as a string, list or Pandas DataFrame of documents. However, real-life data pipelines are rarely so simple. The model inferencing step is nearly always preceded and succeeded by custom data transformation logic. Thus, the "model" being deployed needs to couple the pretrained tranformer model with this custom transformation logic.

MLflow has a `python_function` model flavor that enables this exact use case - coupling model artifacts with custom data handling code into a single model that can be registered, versioned, shared, access-controlled, and exposed with a common set of APIs (e.g. a `predict()` method) regardless of the underlying libraries or logic being used. This project attempts to simplify the creation of Transformer-based MLflow models by providing templates for common use cases. 

