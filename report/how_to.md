# About this book

## How to use this book

### Navigating tags

Each case is categorized by a collection of relevant tags. A specific tag's cases (such as `2D`) can be navigated through the [🏷 Tags](./_tags/tagsindex.md) index.

### Executing and editing code

Press the {fas}`rocket` button on the toolbar, then the {fas}`play` button to edit and run the code.

```{note}
This might take a while to load after new releases of the book.
```

## How to contribute

### Converting Markdown files to Jupyter notebooks

All V&V cases are tracked as Markdown files in the [\{MyST\}NB](https://myst-nb.readthedocs.io/en/latest/) format, which allows
us to use useful [MyST directives](https://myst-parser.readthedocs.io/en/latest/syntax/roles-and-directives.html) and track the code in the same raw text file.
You can run [`jupytext --sync`](https://jupytext.readthedocs.io/en/latest/using-cli.html) on the `.md` MyST-NB file to convert it to a jupytext notebook.
Running the same command will sync all changes between the files.

You can also download any case as a Jupyter notebook by clicking the {fas}`download` button on the toolbar and selecting the {fas}`code` `.ipynb` option.

### Adding a case

1. Create a [Jupyter notebook](https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/what_is_jupyter.html) for editing the case.

2. Run [`jupytext --set-formats ipynb,myst`](https://jupytext.readthedocs.io/en/latest/using-cli.html) on the `.ipynb` notebook file. This creates a ["paired"](https://jupytext.readthedocs.io/en/latest/paired-notebooks.html) Markdown file which we will be tracking on git.

3. Add the path to the case to the table of contents located at `./report/_toc.yml` under the appropriate chapter.

    ```{note}
    The files in the table of contents lack an extension. This is useful for testing since jupyter-book will prioritize `.ipynb` files over `.md` files when building the book.
    ```

You can build the book locally by running `jupyter-book build ./report`.

#### Coding Guidelines

TODO

#### Final Touches

1. Add category tags by using the [\{tags\}](https://sphinx-tags.readthedocs.io/en/latest/quickstart.html#usage) directive below the case title as follows:
    <!-- the "a" in "{tags}" is a homoglyph of the regular a, so that this page does not appear in the tag index -->
    ``````md
    ```{tаgs} tag1, tag2
    ```
    ``````

2. Add ["hide"](https://myst-nb.readthedocs.io/en/latest/render/hiding.html) tags to the code cells in the case where appropriate. Ideal visible outputs only contain figures.
3. Format the code using [black](https://pypi.org/project/black/) on the notebook file as follows `black notebook.ipynb`.

    ```{note}
    Jupyter notebook support for black can be installed with `pip install black[jupyter]`.
    ```
