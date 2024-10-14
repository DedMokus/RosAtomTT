import catboost
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from nltk.corpus import stopwords
import re
import nltk
from nltk.tokenize import word_tokenize
import string
import os

from django.shortcuts import render


class Test_task():

    model_ctb_pos = catboost.CatBoostClassifier()
    model_ctb_neg = catboost.CatBoostClassifier()
    model_ctb_r = catboost.CatBoostRegressor()
    tokenizer = None
    model_BERT = None
    device = torch.device("cpu")


    @classmethod
    def init(self):
        nltk.download('stopwords')
        nltk.download('punkt_tab')
        nltk.download('wordnet')
        self.model_ctb_r.load_model("test_task/model")
        self.model_ctb_neg.load_model("test_task/model_neg")
        self.model_ctb_pos.load_model("test_task/model_pos")
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        self.model_BERT = AutoModel.from_pretrained("google-bert/bert-base-uncased")
        return "Done"

    @classmethod
    def tokenize(self, x):
        x = re.sub(r'[^\w\s]', '', x)
        stopwords_set = set(stopwords.words('english'))
        wnl = nltk.WordNetLemmatizer()  
        x = ' '.join([wnl.lemmatize(word) for word in word_tokenize(x.lower()) if (word not in stopwords_set) and (word not in string.punctuation)])
        t = self.tokenizer(x, padding=True, truncation=True, return_tensors='pt')
        return t

    @classmethod
    def inference(self, text):
        embeddings = []
        text = self.tokenize(text)

        with torch.no_grad():
            model_output = self.model_BERT(**{k: v.to(self.model_BERT.device) for k, v in text.items()})
            embedding = model_output.last_hidden_state[:, 0, :]
            embedding = torch.nn.functional.normalize(embedding)
        embeddings = embedding[0].cpu().numpy()

        embeddings = np.array(embeddings)

        pos = self.model_ctb_pos.predict(embeddings)
        neg = self.model_ctb_neg.predict(embeddings)

        embeddings = np.append(embeddings, [pos, neg])

        pred = self.model_ctb_r.predict(embeddings)

        pred = round(pred)

        return pred


    @classmethod
    def infer_view(self, request):
        print(request)
        result = None
        if request.method == 'POST':

            input_data = request.POST.get('input_data')

            result = self.inference(input_data)

            return render(request, 'inference.html', {'result': result, "input": input_data})
        else:
            return render(request, 'inference.html', {'result': None})







