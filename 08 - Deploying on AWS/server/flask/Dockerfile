# Pull a newer tensorflow image
FROM tensorflow/tensorflow:2.16.1

# Set the working directory to /app
WORKDIR /app

# Transfer content from current dir to /app in container
ADD . /app

# Install audio libraries
RUN apt-get update && apt-get install -y libsndfile1 libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg

# Install python packages
RUN pip install --upgrade pip
RUN pip install --ignore-installed blinker -r requirements.txt

# Create a dedicated non-root user
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Change ownership of your working directory
RUN chown -R appuser:appgroup /app

# Switch to the non-root user before running uWSGI
USER appuser

# Set environment variables for non-root user write access
ENV NUMBA_CACHE_DIR=/tmp
ENV MPLCONFIGDIR=/tmp

# Start uWSGI using config file
CMD ["uwsgi", "app.ini"]
