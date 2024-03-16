# V-V-report

This is the repo of the [FESTIM V&V report](https://festim-vv-report.readthedocs.io/) built with [Jupyter-book](https://jupyterbook.org/).


## Build locally

First clone the repository:

```
git clone https://github.com/festim-dev/V-V-report
```

Create the correct conda environment with the required dependencies:

```
conda env create -f report/environment.yml
conda activate vv-festim-report-env
```

You can then build the book with:
```
jupyter-book build report
```

To force the rebuild of all pages:
```
jupyter-book build report --all
```