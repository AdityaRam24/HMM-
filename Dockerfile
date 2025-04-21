# Use official slim Python runtime as a parent image
FROM python:3.13-slim

# Set a working directory
WORKDIR /app

# Install build dependencies (if you need to compile any C‑extensions)
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir gunicorn

# Copy your application code
COPY . .

# Expose the port your Flask app runs on
EXPOSE 5000

# Set environment variables
ENV FLASK_ENV=production
ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

# Run the app with Gunicorn
#   - workers=3 is a reasonable default; adjust per‑CPU if you like
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "3", "app:app"]
