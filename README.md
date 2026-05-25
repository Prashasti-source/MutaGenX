# MutaGenX

![Python](https://img.shields.io/badge/Python-3.x-blue)

[MutaGenX](http://www.csb.iitkgp.ac.in/applications/MutaGenX/index.php) is a machine learning-based framework developed for predicting the pathogenic impact of protein mutations associated with neurodegenerative disorders using sequence-derived and structural descriptors.

MutaGenX integrates protein language model embeddings, physicochemical descriptors, mutation-aware feature extraction, and machine learning classification approaches for mutation pathogenicity prediction and analysis.

---

# Contents

- [Contents](#contents)
- [Requirements and installation](#requirements-and-installation)
- [Basic usage](#basic-usage)
- [Input format](#input-format)
- [Output](#output)
- [Repository structure](#repository-structure)
- [Applications](#applications)
- [Disclaimer](#disclaimer)

---

# Requirements and installation

This software is developed using Python 3.x. Python 3.x is required as the runtime environment.

```bash
git clone https://github.com/Prashasti-source/MutaGenX.git
cd MutaGenX

# create virtual environment (recommended)
python3 -m venv mutagenx_env

# activate environment
source mutagenx_env/bin/activate

# install dependencies
pip install -r requirements.txt
```

---

# External Dependencies

MutaGenX utilizes the following external resources:

- ESM2 protein language models
- IUPred2A disorder prediction framework

Install ESM2:

```bash
pip install fair-esm
```

Install IUPred2A separately from its official repository.

---

# Basic usage

Users can predict mutation pathogenicity using wild-type and mutant FASTA sequence inputs.

```bash
python3 mutagenx/MutaGenX.py \
-i data/wildtype_1.fasta \
-m data/mutated_1.fasta \
-o outputs/results.csv
```

---

# Input format

## Wild-type FASTA example

```fasta
>WT
MASNDYTQQATQSYNQDQNYSGYQQQQQQSYGQQQSYNPPQGYGQQNQYNS
```

## Mutant FASTA example

```fasta
>MUT
MASNDYTQQATQSYNQDQNYSGYQQQQQQSYGATQSYNPPSGYGQQNQYNS
```

---

# Output

The prediction pipeline generates:

- Mutation prediction labels
- Pathogenicity scores
- Feature matrices
- Prediction summaries

Output files are automatically generated inside the `outputs/` directory.

---

# Repository structure

```text
MutaGenX/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── mutagenx/
│   ├── MutaGenX.py
│   └── train.py
│
├── models/
│   ├── trained_model.pkl
│   └── feature_columns.pkl
│
├── data/
│   ├── wildtype_1.fasta
│   └── mutated_1.fasta
│
├── scripts/
│   ├── run_prediction.sh
│   └── single_prediction.sh
│
├── outputs/
│
└── docs/
    └── user_manual.md
```

---

# Applications

MutaGenX can be used for:

- Neurodegenerative mutation analysis
- Computational pathogenicity prediction
- Protein variant prioritization
- Mutation screening studies
- Biomolecular sequence analytics

---

# Disclaimer

MutaGenX is currently under active development.

Some benchmark datasets, prediction outputs, and extended analyses associated with the corresponding study are not publicly released yet.

---

# Web Server

The MutaGenX web server is available at:

http://www.csb.iitkgp.ac.in/applications/MutaGenX/index.php

---

# Author

Dr. Prashasti Sinha

Researcher in Computational Biophysics and Bioinformatics
