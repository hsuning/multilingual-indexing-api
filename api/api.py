from fastapi import FastAPI
from pydantic import BaseModel

import joblib
from uvicorn import run

import sys, os
sys.path.insert(0, '.')
from MultilingualSentencesIndexing import *

# ## API
# Initializing FastAPI
app = FastAPI()

with open('multilingual_indexing_model.pkl', 'rb') as f:
    model = joblib.load(f)

# Reload preprocessor and encoder
model.preprocessor = hub.KerasLayer(
    hub.load('https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-preprocess/2')
    )
model.encoder = hub.KerasLayer(
    hub.load('https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base/1')
)    

class Data(BaseModel):
    query: str
    top_n: int

# Defining a test root path and message
@app.get('/')
def root():
    return {'message': 'Hello friends!'}


# Defining the indextion endpoint
# With data validation: make use of the Query data model 
# to ensure the request is with 2 features and all features match its data type
@app.post('/predict')
async def predict(data: Data):
    # Converting input data into Pandas DataFrame
    data = data.dict()
    # Getting the prediction from the Logistic Regression model
    pred = model.get_top_n_faqs(data['query'], data['top_n'])
    return pred

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5555))
    run(app, host="0.0.0.0", port=port)
