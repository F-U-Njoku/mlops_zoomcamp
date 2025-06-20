FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.bin .
COPY starter.py .

RUN mkdir -p output/yellow_tripdata

CMD ["python", "starter.py", "2023", "05"]
