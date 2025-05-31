FROM tensorflow/tensorflow:latest

WORKDIR /app

COPY . .
COPY requirements.txt .
RUN pip install -r requirements_docker.txt

VOLUME [ "/app"]
EXPOSE 5000

# to build this container, use:
# docker build -t my_tf_container .

# to run this container, use:
# docker run -it -v C:/full/path/to/models:/app/models -p 5000:5000 my_tf_container /bin/bash

# this will start a bash shell in the container
# to run the application, use:
# python3 project.py