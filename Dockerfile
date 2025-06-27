# Use an official Python runtime as the base image
FROM python:3.11-slim
# Set working directory
WORKDIR /app
# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc git curl unzip libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* 

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set Poetry environment variables
ENV PATH="/root/.local/bin:$PATH"

# Copy the dependency specification files to leverage Docker cache
COPY pyproject.toml poetry.lock* ./

# Install dependencies using Poetry
RUN poetry config virtualenvs.create false && \
    poetry lock  && \
    poetry install --no-interaction --no-ansi
# Copy the rest of your application code
COPY . .

# Expose port (adjust if needed)
EXPOSE 7860

#CONFIG TEMP DIRECTORY
RUN mkdir -p /tmp/hf_cache
ENV XDG_CACHE_HOME=/tmp/hf_cache

# Run the application using Poetry's runner; adjust the command if your app entrypoint differs.
CMD ["sh", "-c", "cd app && python -m litestar run --host 0.0.0.0 --port 7860"]