# V-V-report

This is the repo of the [FESTIM V&V report](https://festim-vv-report.readthedocs.io/) built with [Jupyter-book](https://jupyterbook.org/).

## Build locally

First clone the repository:

```bash
git clone https://github.com/festim-dev/V-V-report
```

Create the correct conda environment with the required dependencies:

```bash
conda env create -f environment.yml
conda activate vv-festim-report-env
```

You can then build the book with:

```bash
jupyter-book build report
```

To force the rebuild of all pages:

```bash
jupyter-book build report --all
```

## Contributing

Run [`jupytext --sync`](https://jupytext.readthedocs.io/en/latest/using-cli.html) on a case's markdown file to generate a matching [Jupyter notebook](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html). Run [`jupytext --sync`](https://jupytext.readthedocs.io/en/latest/using-cli.html) again to sync the changes to the markdown file.

See [this page](./report/how_to.md/#how-to-contribute) for more instructions.
