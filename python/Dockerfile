FROM python:3.11.7-slim-bookworm

WORKDIR /app/db

COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . .

RUN mkdir /app/db/data

CMD ["python", "embedserver.py"]
