FROM python:3.6

EXPOSE 5000

WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt


COPY models /app/models
COPY resources /app/resources
COPY db.py /app
COPY app.py /app
CMD python app.py
