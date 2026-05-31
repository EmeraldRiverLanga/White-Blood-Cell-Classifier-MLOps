# Use a slim Python base image — smaller than the full image, enough for us.
FROM python:3.11-slim

# Prevent Python from writing .pyc files and buffering stdout (cleaner logs).
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Where the model lives inside the container. The app reads this env var.
ENV MODEL_DIR=/app/models

# Set the working directory inside the container.
WORKDIR /app

# Install dependencies first, separately from the code. Docker caches this
# layer, so changing app code later does not re-install everything.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code, the Streamlit config, and the trained model.
COPY app/ ./app/
COPY .streamlit/ ./.streamlit/
COPY models/ ./models/

# Streamlit's default port.
EXPOSE 8501

# Launch the app. Streamlit reads address/port from .streamlit/config.toml.
CMD ["streamlit", "run", "app/app.py"]