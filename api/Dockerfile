FROM tiangolo/uvicorn-gunicorn:python3.10-slim

# Author info
LABEL maintainer="Hsuning Chang"

# Make working directories
RUN  mkdir -p  /api
WORKDIR  /api

# Upgrade pip / apt-get
RUN pip install --no-cache-dir -U pip

# Copy every file in the source folder to the created working directory
COPY . .

RUN pip install -r requirements.txt

# Exposing the appropriate port on the container
EXPOSE 5555

# Run the python application
CMD ["python", "api.py"]