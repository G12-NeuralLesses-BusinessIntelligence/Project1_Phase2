from joblib import load
import pandas as pd
from nltk.tokenize import TweetTokenizer

def tokenizer(text):
    return TweetTokenizer().tokenize(text)
class Model:
    def __init__(self):
        self.pipeline = load('assets/text_classifier_base.joblib')
        

    def make_predictions(self, lista_msgs):
        tokenizer = lambda text: tokenizer(text)
        results = self.pipeline['model'].predict(self.pipeline['tfidf'].transform(pd.Series(lista_msgs)))
        return results

