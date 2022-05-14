FROM continuumio/miniconda3:4.10.3p1

COPY work/environment.yml /conda_temp/environment.yml

# Install dependencies
RUN conda env create -f /conda_temp/environment.yml

# OpenCV dependency
RUN apt-get update && apt-get install libgl1 -y


# Make RUN commands use the new environment:
# SHELL ["conda", "run", "--no-capture-output", "-n", "cab420-env", "/bin/bash", "-c"]

WORKDIR /work