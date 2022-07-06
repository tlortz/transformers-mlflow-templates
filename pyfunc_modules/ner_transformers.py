"""
Template for taking a Huggingface Transformer Named Entity Recognition pipeline and registering it as an MLflow model.
"""

import mlflow
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class TransformerNERModel(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self._model = model
        
    def _extract_entities(self, ner_list):
        """
        Helper function for extracting detected named entities into a list of (entity type, word chunk) tuples
        """
        if len(ner_list) > 0:
            return [(e["entity"], e["word"]) for e in ner_list]
        else:
            return []
    
    def predict(self, df):
        """
        Inference logic that uses a Huggingface Transformer Named Entity Recognition pipeline and extracts named entities from one or more documents.

        Parameters
        ----------
        df: pandas.DataFrame
            data frame containing at least two columns: 
                'id'(Integer) - unique identifier for each row in the dataframe
                'content'(String) - text or document objects
            any other columns will be dropped in the returned results

        Returns
        -------
        pandas.DataFrame
            data frame containing three columns:
                'id'(Integer) - unique identifier for each row in the dataframe
                'content' (String) - original text or document objects
                'ner_extracted'(List((String, String))) - list of (entity type, word chunk) pairs resulting from the NER model applied to the content
        """
        texts = df.content.values.tolist()
        ids = df.id.values.tolist()
        text_ner = self._model(texts, batch_size=2)
        ner_extracted = list(map(self._extract_entities, text_ner))
        df_with_ents = pd.DataFrame({"id": ids, "content": texts, "ner_extracted": ner_extracted})
        return df_with_ents

def _load_pyfunc(data_path):
    """
    Required PyFunc custom loader module, following the second option in https://www.mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#workflows
    Note that although the Huggingface tokenizer and model instantiation calls can infer a web URL from a model card path, e.g. "dslim/bert-base-NER",
    MLflow Pyfunc loader modules treat the data_path argument as being in the local file system, i.e. in S3, ADLS or DBFS
    """
    device = 0 if torch.cuda.is_available() else -1
    tokenizer = AutoTokenizer.from_pretrained(data_path, padding=True)
    model = AutoModelForTokenClassification.from_pretrained(data_path)
    nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=device)
    return TransformerNERModel(nlp)