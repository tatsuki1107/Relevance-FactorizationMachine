FROM python:3.9-slim-buster

WORKDIR /app

RUN pip install --upgrade poetry && poetry config virtualenvs.create false

COPY poetry.lock pyproject.toml ./

RUN poetry install --no-dev
