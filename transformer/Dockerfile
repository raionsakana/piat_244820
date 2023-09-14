FROM python:3.10
LABEL authors="Patryk Janowski, 244820"

RUN pip install tensorflow keras && mkdir app

WORKDIR app
COPY vocab vocab
COPY weights weights
COPY bajki.txt bajki.txt
COPY utils.py utils.py
COPY app.py app.py

ENTRYPOINT ["top", "-b"]