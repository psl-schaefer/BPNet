
### R Environment

- Use mamba (or conda) to create the corresponding environments

```bash
mamba env create -f envs/r_env.yml
```

- However, on Apple Silicon we have to use a little trick

- See https://stackoverflow.com/questions/76470802/cannot-install-r-bioconductor-packages-via-conda

```bash
CONDA_SUBDIR=osx-64 mamba env create -f envs/r_env.yml
```

- Remove it with

```bash
mamba env remove --name r-env-bpnet
```

### Python Environment

- Use mamba (or conda) to create the corresponding environments

```bash
mamba env create -f envs/py_env.yml
```

- Remove it with

```bash
mamba env remove --name py-env-bpnet
```