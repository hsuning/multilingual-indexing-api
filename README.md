# Algomo

## Multilingual Indexing API

Build an API with multilingual (100+ languages) sentence grouping model (https://github.com/hsuning/multilingual-sentences-grouping) for question indexing.


### Folder Structure
    .
    ├── api                    # For dockerizing
    │    ├── api.py             # Python codes for api, py version of the Task-3 in /notebooks 
    │    ├── MultilingualSentencesIndexing.py   # Class for indexing model, py version of Multilingual_Sentences_Indexing_Model.ipynb in /notebooks
    │    ├── multilingual_indexing_model.pkl    # Indexing model with data stored in pkl file
    │    ├── requirements.txt   # Python libraries used by Dockerfile
    │    └── Dockerfile         # Dockerfile for docker image building
    ├── data                    # Data generated by notebooks
    ├── installation            # Files for development environment installation
    ├── notebooks               # Codes with solution explainations
    ├── LICENSE
    └── README.md

### Built With
This section list all frameworks/libraries used.
- fastapi==0.87.0
- joblib==1.1.0
- numpy==1.23.3
- pandas==1.4.4
- pydantic==1.10.2
- scikit_learn==1.1.3
- tensorflow_hub==0.12.0
- tensorflow_text==2.10.0
- uvicorn==0.20.0
- universal-sentence-encoder-cmlm/multilingual-base: <https://tfhub.dev/google/universal-sentence-encoder-cmlm/multilingual-base/1>

<!-- GETTING STARTED -->
## Getting Started

### Run the Indexing-API with Docker Playground on the Cloud
You can first test the API without installing any things. The only thing you need is a web browser. In **Test-model-on-cloud.pdf**, you can find the same instruction with **screenshots**.

1. Open a web browser and go to <https://labs.play-with-docker.com/>

2. Login with your docker account and click on the start button

4. Add a new instance

5. Input the command below to pull the image from my Docker Hub (around 1 minute):
```
docker pull hsuningchang/indexing-linux:latest
```
> The link to my Docker Hub is <https://hub.docker.com/r/hsuningchang/indexing-linux.> I will delete it after your testing.

6. Once the image is downloaded, input the command below and wait for server to start (around 1.5 minute as it will download modules from Tensorflow Hub)
```
docker run -p 5555:5555 hsuningchang/indexing-linux:latest
```

7. The following message indicates that the server starts up:
```
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:5555 (Press CTRL+C to quit)
```

8. Click on the “OPEN PORT” button on the top and input “5555”. Don’t forget to allow the pop-ups in your browser.
    
9. A page with “Hollo friends!” message shows. The URL of the page would be something similar to <http://ip172-18-0-4-ce37ldm3tccg009angdg-5555.direct.labs.play-with-docker.com/>
    
10. Then add another new instance
    
11. Input the command below for testing. Don’t forget to modify the URL part (red colour). The response time is generally lower than 3 seconds.
```
curl --request POST \
--header 'Content-Type: application/json' \
--data '{"query":"credit card", "top_n":20}' \
--url http://ip172-18-0-9-ce38kgm0qau0009v79o0-5555.direct.labs.play-with-docker.com/predict
```

> Please feel free to modify the –data part to test with other query or get more results.


12. The output result will be :

```
{"0":{"question":"How to close my account?","Ranking":1.0,"FAQ_id":40,"locale":"en","market":"en-de"},"1":{"question":"How to close my account?","Ranking":1.0,"FAQ_id":40,"locale":"en","market":"en-es"},"2":{"question":"How to close my account?","Ranking":1.0,"FAQ_id":40,"locale":"en","market":"en-eu"},"3":{"question":"How to close my account?","Ranking":1.0,"FAQ_id":40,"locale":"en","market":"en-at"},"4":{"question":"How to close my account?","Ranking":1.0,"FAQ_id":40,"locale":"en","market":"en-fr"},"5":{"question":"How to close my account?","Ranking":1.0,"FAQ_id":40,"locale":"en","market":"en-it"},"6":{"question":"Wie kann ich mein Konto schließen?","Ranking":2.0,"FAQ_id":40,"locale":"de","market":"de-at"},"7":{"question":"Wie kann ich mein Konto schließen?","Ranking":2.0,"FAQ_id":40,"locale":"de","market":"de-de"},"8":{"question":"How to protect my account?","Ranking":3.0,"FAQ_id":90,"locale":"en","market":"en-it"},"9":{"question":"How to protect my account?","Ranking":3.0,"FAQ_id":90,"locale":"en","market":"en-de"}}
```

### Installation
The installation should be on MacBook M1 or M2.

In this section, we will create a new Conda environment and install all the necessary packages. At the end of the installation, you can follow the instruction to test the API.

1. Open a terminal and go to the folder **/installation**:
```sh
cd Hsuning_Chang_ml_eng/installation
```

2. Create an environment with ```environment.yaml``` file and activate it:

> Please note that you can change the environment name in this file.

```sh
  conda env create --file=environment.yaml
  conda activate multilingual-indexing
```

3. Update the pip and install all the packages with ```requirements.txt```:

```sh
  pip install --no-cache-dir -U pip
  pip install -r requirements.txt
```

4. To verify the installation, go to **/Code** folder and run the command below.

> It takes around 1 minute to start as it has to load TensorFlow model from Tensorflow Hub.

```sh
  cd Code
  python api.py
```

5. You might find some warning messages similar to the messages below, please ignore them:

```
2022-11-29 14:24:28.153454: W tensorflow/core/platform/profile_utils/cpu_utils.cc:128] Failed to get CPU frequency: 0 Hz
WARNING:tensorflow:Please fix your imports. Module tensorflow.python.training.tracking.data_structures has been moved to tensorflow.python.trackable.data_structures. The old module will be deleted in version 2.11.
```

6. Once the server starts up, you will see the following message:

```
INFO:     Started server process [84934]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:5555 (Press CTRL+C to quit)
```

7. Then you can open another Terminal and copy the command below for testing:

```
curl --request POST \
--header 'Content-Type: application/json' \
--data '{"query":"how to close an account", "top_n":10}' \
--url http://0.0.0.0:5555/predict
```

8. The result will be:

```
{"0":{"question":"How to close my account?","Ranking":1.0,"FAQ_id":40,"locale":"en","market":"en-de"},"1":{"question":"How to close my account?","Ranking":1.0,"FAQ_id":40,"locale":"en","market":"en-es"},"2":{"question":"How to close my account?","Ranking":1.0,"FAQ_id":40,"locale":"en","market":"en-eu"},"3":{"question":"How to close my account?","Ranking":1.0,"FAQ_id":40,"locale":"en","market":"en-at"},"4":{"question":"How to close my account?","Ranking":1.0,"FAQ_id":40,"locale":"en","market":"en-fr"},"5":{"question":"How to close my account?","Ranking":1.0,"FAQ_id":40,"locale":"en","market":"en-it"},"6":{"question":"Wie kann ich mein Konto schließen?","Ranking":2.0,"FAQ_id":40,"locale":"de","market":"de-at"},"7":{"question":"Wie kann ich mein Konto schließen?","Ranking":2.0,"FAQ_id":40,"locale":"de","market":"de-de"},"8":{"question":"How to protect my account?","Ranking":3.0,"FAQ_id":90,"locale":"en","market":"en-it"},"9":{"question":"How to protect my account?","Ranking":3.0,"FAQ_id":90,"locale":"en","market":"en-de"}}
```

9-You can modify the ```--data``` part to test with other queries, for example:

```
--data '{"query":"open an account", "top_n":3}' 

--data '{"query":"credit card", "top_n":15}' 
```

### Dockerizing
The new MacBook with Apple-designed systems on a chip (M1 and M2) uses the ARM64 architecture. When building a docker image, it is important to support multiple platforms - different architectures and different operating systems.

After countless attempts, I unfortunately have never been able to run a docker image with tensorflow and tensorflow-text on my Macbook Pro M2. Especially, it seems tensorflow-text don't support ARM64, hence, the installation could be complicated. However, I found an alternative.

The final solution used is to build a ```linux/arm64``` version of docker imag and test it on the cloud.

**How to build a docker image**  
1- Open a terminal  
2- Go to the /Code folder and run api.py file

> You will need a docker hub account  
> username = your docker username  
> image = the name of image

```sh
docker buildx build --push --platform linux/arm64 -t <username>/<image>:latest .
```

### API

#### Run in Jupyter Notebook:
1. Run the whole notebook  
2. Open a terminal and input the command below

```
curl --request POST \
--header 'Content-Type: application/json' \
--data '{"query":"how to close an account", "top_n":10}' \
--url http://0.0.0.0:5555/predict
```

3. You can modify the ```--data``` section by using different sentences or number of topics

#### Run Using Terminal
1. Open a terminal  
2. Activate the conda environment, go to the /Code folder and run api.py file

```
conda activate multilingual-indexing
cd Code
python api.py
```

3. open another terminal and input the command below
```
curl --request POST \
--header 'Content-Type: application/json' \
--data '{"query":"how to close an account", "top_n":10}' \
--url http://0.0.0.0:5555/predict
```

3- you can modify the ```--data``` section by using different sentences or number of topics

<!-- LICENSE -->

## Debugging

### OSError: SavedModel File Does Not Exist
When runing the code ```python api.py```, if you found the error below:

```
OSError: SavedModel file does not exist at: /var/folders/59/sjwssnws45n7dw_8b5jxsbc00000gn/T/tfhub_modules/8e75887695ac632ead11c556d4a6d45194718ffb/{saved_model.pbtxt|saved_model.pb}
```

Open a terminal and input:

```
% open $TMPDIR
```

A folder will be opened. Please delete the subfolder /tfhub_modules and rerun the code

<!-- LICENSE -->

## License
Distributed under the MIT License. See `LICENSE.txt` for more information.
