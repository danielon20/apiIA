from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel
import uvicorn
from keras.preprocessing.text import Tokenizer, tokenizer_from_json
import json
from nltk.stem import SnowballStemmer
import regex as re
import numpy as np

from keras.utils import pad_sequences
from tensorflow.keras.models import load_model


###LIMPIEZA DE TEXTO
punctuations = "¡!#$%&'()*+,-./:;<=>¿?@[\]^_`{|}~"

def read_txt(filename):
    list = []
    with open(filename, 'r', encoding='utf-8') as f:
        data = f.readlines()
        for line in data:
            list.append(str(line).replace('\n', ''))
    return list

stopwords = read_txt('english_stopwords.txt')

stemmer = SnowballStemmer('english')


def clean_accents(tweet):
    tweet = re.sub(r"[àáâãäå]", "a", tweet)
    tweet = re.sub(r"ç", "c", tweet)
    tweet = re.sub(r"[èéêë]", "e", tweet)
    tweet = re.sub(r"[ìíîï]", "i", tweet)
    tweet = re.sub(r"[òóôõö]", "o", tweet)
    tweet = re.sub(r"[ùúûü]", "u", tweet)
    tweet = re.sub(r"[ýÿ]", "y", tweet)

    return tweet

def clean_tweet(tweet, stem = False):
    tweet = tweet.lower().strip()
    tweet = re.sub(r'https?:\/\/\S+', '', tweet)
    tweet = re.sub(r'http?:\/\/\S+', '', tweet)
    tweet = re.sub(r'www?:\/\/\S+', '', tweet)
    tweet = re.sub(r'\s([@#][\w_-]+)', "", tweet)
    tweet = re.sub(r"\n", " ", tweet)
    tweet = clean_accents(tweet)
    tweet = re.sub(r"\b(a*ha+h[ha]*|o?l+o+l+[ol]*|x+d+[x*d*]*|a*ja+[j+a+]+)\b", "<risas>", tweet)
    for symbol in punctuations:
        tweet = tweet.replace(symbol, "")
    tokens = []
    for token in tweet.strip().split():
        if token not in punctuations and token not in stopwords:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)
###FIN DE LIMPIEZA DE TEXTO

def f1_score(y_true, y_pred):
    
    # Se cuenta las muestras positivas.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # Si no hay muestras verdaderas, fije la puntuación F1 en 0.
    if c3 == 0.0:
        return 0.0

    # ¿Cuántos items seleccionados son relevantes?
    precision = c1 / c2

    # ¿Cuántos items relevantes son seleccionados?
    recall = c1 / c3
    
    # Calculo de f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

with open('tokenizer.json') as f:
    data = json.load(f)
    tok = tokenizer_from_json(data)

model = load_model('./model/model.keras',custom_objects={"f1_score": f1_score})

app = FastAPI()

origins = [ "*" ]

class TextTweet(BaseModel):
    texto: str
    
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def index():
    return "prueba"

@app.post("/prediction")
def predict(twet: TextTweet):
    proff_clean = clean_tweet(twet.texto)
    seq = tok.texts_to_sequences([proff_clean])
    seq_matrix = pad_sequences(seq)
    test_sequence_matrix = np.lib.pad(seq_matrix, ((0,0),(31 - seq_matrix.shape[1],0)), 'constant', constant_values=(0))

    y2_pred_prob = model.predict(test_sequence_matrix) 
    y2_pred_lab = np.where(y2_pred_prob > 0.5, 1, 0).flatten()
    
    valor = str(y2_pred_lab[0])
    return {"resultado" : valor}


#if __name__ == "__main__":
    #uvicorn.run(app, host="0.0.0.0", port=8000)
#app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
