FROM python:3.11.9

ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN python -m pip install -r requirements.txt

WORKDIR /app

COPY . /app

EXPOSE 5000

CMD ["streamlit", "run", "app.py"]
