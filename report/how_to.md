# User Guide

## How to use this book

### Navigating tags

Each case is tagged and a tag's cases can be navigated through the [🏷 Tags](./_tags/tagsindex.md) index.

### Executing and editing code

Press the ![Live Code button]([./how_to_media/live_code_button.png](https://fontawesome.com/icons/rocket?f=classic&s=solid)) Live Code button on the toolbar to edit and run the code.

```{note}
This might take a while to load after new releases of the book.
```

## Converting Markdown files to Jupyter notebooks

All V&V cases are tracked as Markdown files in the [\{MyST\}NB](https://myst-nb.readthedocs.io/en/latest/) format, which allows
us to use useful [MyST directives](https://myst-parser.readthedocs.io/en/latest/syntax/roles-and-directives.html) and track the code in the same raw text file.
You can run [`jupytext --sync`](https://jupytext.readthedocs.io/en/latest/using-cli.html) on the `.md` MyST-NB file to convert it to a jupytext notebook.
Running the same command will sync all changes between the files.

## Adding a case

1. Create a [Jupyter notebook](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html) for editing the case.

2. Run [`jupytext --set-formats ipynb,myst`](https://jupytext.readthedocs.io/en/latest/using-cli.html) on the `.ipynb` notebook file. This creates a ["paired"](https://jupytext.readthedocs.io/en/latest/paired-notebooks.html) Markdown file which we will be tracking on git.

3. Add to ToC TODO

### Coding Guidelines

TODO

### Final Touches

1. Add category tags by using the [\{tags\}](https://sphinx-tags.readthedocs.io/en/latest/quickstart.html#usage) directive below the case title:

    ``````md
    ```{tags} tag1, tag2
    ```
    ``````

2. Add ["hide"](https://myst-nb.readthedocs.io/en/latest/render/hiding.html) tags to the code cells in the case where appropriate. Ideal visible outputs only contain figures.
3. Format the code using [black](https://pypi.org/project/black/) on the notebook file as follows `black notebook.ipynb`.

    ```{note}
    Jupyter notebook support for black can be installed with `pip install black[jupyter]`.
    ```
