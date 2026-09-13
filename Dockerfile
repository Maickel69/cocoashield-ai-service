FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy microservice code
COPY . .

# Expose port and configure environment
ENV PORT=8000
EXPOSE 8000

CMD ["python", "main.py"]
