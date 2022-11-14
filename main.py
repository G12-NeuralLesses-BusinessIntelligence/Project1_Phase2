from typing import Optional
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import pandas as pd
import numpy as np
from joblib import load,  dump
import json
from PredictionModel import Model

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from nltk.tokenize import TweetTokenizer

from PredictionModel import tokenizer 

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/")
def read_root():
   return {"Hello": "World"}

@app.get("/page", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("page.html", {"request": request})



@app.post("/page", response_class=HTMLResponse)
async def read_item(request: Request, name: str = Form(...), comment: str = Form(...)):
   pipeline = load('assets/text_classifier.joblib')
   result = pipeline['model'].predict(pipeline['tfidf'].transform(pd.Series([comment])))
   print(result)
   return templates.TemplateResponse("page.html", {"request": request,"result":result})


@app.post("/predict")
def make_predictions(lista_msgs: list ):

   pipeline = load('assets/text_classifier.joblib')
   results = pipeline['model'].predict(pipeline['tfidf'].transform(pd.Series(lista_msgs)))
   return {"results": pd.DataFrame(results).to_dict()}
