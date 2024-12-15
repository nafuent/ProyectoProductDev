FROM python:3.12-slim

WORKDIR /code
ENV FLASK_APP api_flask.py
ENV FLASK_RUN_HOST 0.0.0.0


COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY house_price_model.h5 house_price_model.h5
COPY encoder.pkl encoder.pkl
COPY scaler.pkl scaler.pkl
COPY api_flask.py api_flask.py

EXPOSE 5000

CMD [ "flask" ,"run"]