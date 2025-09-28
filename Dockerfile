FROM python:3.11.13-slim-buster

RUN apt update -y && apt install awscli -y

WORKDIR /app

COPY . /app

RUN pip install requirements.txt

CMD ["python", "app.py"]