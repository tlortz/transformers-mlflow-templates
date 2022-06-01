class TransformerNERModel(mlflow.pyfunc.PythonModel):
    # expects an input dataframe with at least the columns "content" and "id"

    import torch
    import pandas as pd
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    
    def __init__(self, model):
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._model = model

    def _load_pyfunc(model_name_or_path="dslim/bert-base-NER"):
        # web-based model is "dslim/bert-base-NER". This can also be a file path in S3 or ADLS
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding=True)
        model = AutoModelForTokenClassification.from_pretrained(model_name_or_path)
        nlp = pipeline("ner", model=model, tokenizer=tokenizer)
        return TransformerNERModel(nlp)

    def _extract_entities(ner_list):
        if len(ner_list) > 0:
            return [(e["entity"], e["word"]) for e in ner_list]
        else:
            return []
    
    def predict(self, df):
        texts = df.content.values.tolist()
        ids = df.id.values.tolist()
        model = self._model.to(self._device)
        text_ner = model(texts, batch_size=2)
        ner_extracted = list(map(self._extract_entities, text_ner))
        df_with_ents = pd.DataFrame({"id": ids, "content": texts, "ner_extracted": ner_extracted})
        return df_with_ents


    
