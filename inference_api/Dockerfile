FROM python:3.9

RUN apt-get update

WORKDIR /code

COPY inference_api/main.py /code/main.py
COPY inference_api/utils.py /code/utils.py
COPY inference_api/requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

EXPOSE 80

CMD uvicorn --host 0.0.0.0 --port 80 --workers 1 main:app