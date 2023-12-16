# Use the official Python image as the base image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container and install the required dependencies
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Copy the Flask application code into the container
COPY /.env .
COPY /run.py .
COPY /app ./app
COPY /Prediksi ./Prediksi
COPY /imageProses ./imageProses

# add environment variables
ENV MYSQL_HOST=10.1.1.13
ENV MYSQL_USER=root
ENV MYSQL_PORT=3306
ENV MYSQL_PASSWORD=abogoboga
ENV MYSQL_DB=peternakan_kambing_cerdas
# Expose the port your Flask app will run on
EXPOSE 3001

# Define the command to run your Flask application
# CMD ["python", "run.py"]
#use gunicorn
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:3001", "run:app"]
