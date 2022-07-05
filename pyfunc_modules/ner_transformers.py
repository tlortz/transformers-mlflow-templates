import mlflow
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

class TransformerNERModel(mlflow.pyfunc.PythonModel):
    # expects an input dataframe with at least the columns "content" and "id"
  def __init__(self, model):
    self._model = model

  def _extract_entities(self, ner_list):
    if len(ner_list) > 0:
      return [(e["entity"], e["word"]) for e in ner_list]
    else:
      return []
    
  def predict(self, df):
    texts = df.content.values.tolist()
    ids = df.id.values.tolist()
    text_ner = self._model(texts, batch_size=2)
    ner_extracted = list(map(self._extract_entities, text_ner))
    df_with_ents = pd.DataFrame({"id": ids, "content": texts, "ner_extracted": ner_extracted})
    return df_with_ents

def _load_pyfunc(data_path):
  # web-based model is "dslim/bert-base-NER". This can also be a file path in S3 or ADLS
  device = 0 if torch.cuda.is_available() else -1
  tokenizer = AutoTokenizer.from_pretrained(data_path, padding=True)
  model = AutoModelForTokenClassification.from_pretrained(data_path)
  nlp = pipeline("ner", model=model, tokenizer=tokenizer, device=device)
  return TransformerNERModel(nlp)
    
