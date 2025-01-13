# Dockerfile for AI Juris
FROM python:3.12-slim

WORKDIR /app

# Copy project files
COPY ../. /app/

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libffi-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

# Install the project dependencies using Poetry
RUN poetry install --no-root

# Command to execute the Python script
CMD ["poetry", "run", "python", "app/evaluation.py", "Input your prompt"]
