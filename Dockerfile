FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Set environment path
ENV PYTHONPATH=/app

# Default execution
CMD ["python", "inference.py"]
