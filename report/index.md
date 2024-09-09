---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.2
kernelspec:
  display_name: vv-festim-report-env
  language: python
  name: python3
---

# FESTIM V&V book

Authors: Remi Delaporte-Mathurin, Jair Santana

Welcome to the FESTIM Verification and Validation (V&V) book! This book aims to provide a comprehensive overview of the verification and validation processes conducted for the FESTIM code. Through a series of verification and validation cases, we aim to demonstrate the accuracy, reliability, and applicability of FESTIM for simulating hydrogen transport phenomena.


`````{admonition}  How to cite this book
:class: tip
  Remi Delaporte-Mathurin and Jair Santana, *FESTIM V&V Book*, 2024, [https://dspace.mit.edu/handle/1721.1/156690](https://dspace.mit.edu/handle/1721.1/156690)

`````

## About this Book

This book has been generated using [Jupyter Book](https://jupyterbook.org/), an open-source tool for building interactive and collaborative online books using Jupyter Notebooks. Jupyter Book enables us to seamlessly integrate code, text, equations, and visualisations, providing an interactive and transparent platform for presenting our verification and validation results. By utilising Jupyter Book, we aim to ensure the reproducibility of our results and provide readers with the ability to explore and interact with the code and data presented in this book.

One key distinction of this book compared to traditional reports, such as the TMAP7 report, is our emphasis on providing complete transparency and reproducibility. While other reports may make the code available, many post-processing steps are often missing, making it challenging for readers to fully understand and reproduce the results. In contrast, we strive to provide not only the code but also detailed descriptions of post-processing procedures, ensuring that readers have all the necessary information to replicate our findings accurately.

Furthermore, all the code cells in this book are run automatically at every push to the book repository. This ensures that the results presented are always up-to-date and reproducible. Additionally, the book is compiled on the fly with ReadTheDocs, allowing readers to access the latest version of the book online with ease. Previous versions of the book are also available.

```{code-cell} ipython3
import festim

print(f"FESTIM version: {festim.__version__}")
```

## Feedback and Contributions

We encourage readers to provide comments, suggestions, and corrections to this book by submitting them to the issue tracker. To do so, simply navigate to the relevant page in the tutorial, then click the <img src="https://github.githubassets.com/assets/GitHub-Mark-ea2971cee799.png" height="20"> symbol in the top right corner and select "open issue". Your feedback is invaluable in helping us improve and refine this resource for the benefit of the community.

## Acknowledgments

We would like to express our gratitude to all [contributors](https://github.com/festim-dev/FESTIM/graphs/contributors) who have contributed to the development and improvement of FESTIM. Your dedication and support have been instrumental in advancing the capabilities of FESTIM and making this V&V book possible.

We hope that this book serves as a valuable resource for researchers, practitioners, and enthusiasts interested in hydrogen transport modeling and the verification and validation of computational codes. Thank you for joining us on this journey of exploration and discovery.
