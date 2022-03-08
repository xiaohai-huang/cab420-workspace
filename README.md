# CAB420 Machine Learning Python Environment

`/work` is Jupyter Notebook's working directory. Use bind mount to map local folder to container's `/work`

## Get Started

```bash
git clone https://github.com/xiaohai-huang/cab420-workspace
docker-compose up
```

## Update packages

In container, the environment.yml file should be located inside `work` directory.

```bash
conda env export > environment.yml
```

## Note

The default conda environment is `cab420-env`
