# Introduction

Hydrogen transport modelling stands as a pivotal discipline across various scientific domains, including materials science, energy storage, nuclear fusion, and transportation. Yet, traditional tools within these fields have encountered limitations, often restricted to **1D simulations** and struggling with **multi-material handling**. Additionally, many are ensnared in **outdated programming languages**, hindering collaborative advancements.

In response to these challenges, the Finite Element Simulation of Tritium In Materials (**FESTIM**) emerged. FESTIM was meticulously crafted to transcend these limitations, offering a **simpler, more flexible avenue** for modelling hydrogen transport. Our foremost objective with FESTIM is to **democratise the simulation of hydrogen migration**, fostering ease of use and accessibility. Furthermore, FESTIM integrates heat transfer simulations to enhance temperature accuracy, expanding its utility across diverse applications.

What distinguishes FESTIM is its utilisation of the versatile finite element method, built upon the **FEniCS library**—an open-source platform with a Python interface, thus extending its reach to a broader audience. The open-source nature of FESTIM not only encourages widespread adoption but also facilitates collaborative development. Under the permissive **Apache-2.0 license**, FESTIM invites users to tailor the code to their specific requirements, fostering a thriving community of contributors spanning diverse research institutions and private enterprises.

Inspired by the comprehensive [**Verification and Validation (V&V) report of TMAP7**](https://inldigitallibrary.inl.gov/sites/sti/sti/4215153.pdf){cite}`longhurst_tmap7_2008`, which has set the standard for many years, this book serves as an extensive exploration of FESTIM's capabilities within the context of verification and validation. In contrast to existing tools such as MHIMS, TESSIM, and CRDS, FESTIM's hallmark lies in its **open-source framework** and its adeptness in handling **multidimensional simulations** across diverse materials. Its **user-friendly interface** accommodates scenarios ranging from simple 1D simulations to intricate 3D models, effectively addressing heat transfer and material interface complexities. While alternatives like TMAP8 and ACHLYS offer similar capabilities, they are built upon the finite element code MOOSE, differing from FESTIM's unique architecture.

With applications extending beyond nuclear fusion, FESTIM finds utility in academia and private sectors alike, ranging from experimental data analysis to the design and analysis of hydrogen-interacting components. As FESTIM garners increasing attention, this book serves as a comprehensive report of its verification and validation processes, elucidating why it has emerged as a reliable tool for simulating hydrogen transport.

As the FESTIM community continues to evolve, this report will serve as a living document, continuously updated to reflect advancements in the codebase. Each validation and verification case presented herein is accompanied by detailed descriptions, theoretical backgrounds, and, crucially, all requisite FESTIM and Python code for reproducibility, thereby ensuring transparency and accessibility.

Welcome to the V&V book of FESTIM — a testament to collaborative innovation in the realm of hydrogen transport modelling.

```{tableofcontents}
```
