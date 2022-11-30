from fastapi import FastAPI
from uvicorn import run


import sys, os
sys.path.insert(0, '.')
from MultilingualSentencesIndexing import *

# ## API
# Initializing FastAPI
app = FastAPI()

# Defining a test root path and message
@app.get('/')
def root():
    return {'message': 'Hello friends!'}

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5555))
    run(app, host="0.0.0.0", port=port)
