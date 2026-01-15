# Use a lightweight official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory to /app inside the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose port 5000 for external access
EXPOSE 5000

# Define the command to run your Flask app
CMD ["python", "app.py"]
