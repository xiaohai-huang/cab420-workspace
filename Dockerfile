FROM continuumio/miniconda3:4.10.3p1

COPY work/environment.yml /conda_temp/environment.yml

# Install dependencies
RUN conda env create -f /conda_temp/environment.yml

# Make RUN commands use the new environment:
# SHELL ["conda", "run", "--no-capture-output", "-n", "cab420-env", "/bin/bash", "-c"]

WORKDIR /work

EXPOSE 8888

CMD ["conda", "run", "--no-capture-output", "-n", "cab420-env", "jupyter-lab","--ip=0.0.0.0","--no-browser","--allow-root"]