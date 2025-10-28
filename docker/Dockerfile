FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy the entire project into container
COPY .. /app

# Add to Python path so "import apps" works
ENV PYTHONPATH="/app"

# Install requirements
RUN pip install --upgrade pip && \
    pip install -r /app/apps/requirements.txt

# Train model inside the container (Optional but works)
RUN python /app/apps/train.py

EXPOSE 5000

# Start Flask app using Gunicorn
CMD ["gunicorn", "apps.flask_app:app", "-b", "0.0.0.0:5000", "--workers", "2"]
