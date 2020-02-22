# Datasets
To provide credible evaluation for the performance of our ULM, we catalog a benchmark dataset for Arabic which can also be used for future research benchmark evaluations. The data sets vary in size allowing us to demonstrate the ULM’s abilities to fine tune with little data and achieve high performance. The benchmark data set is summarized in the following table:

  **DATASET** | **RESOURCE** | **# SAMPLES** | **# CLASSES** | **MSA \|\| DIALECT**
------------- | ------------ | ------------- | ------------- | ------------------
HARD | Hotel reviews </br> (www.booking.com) | 93700 | 4 | MSA & Gulf
ASTD | Twitter | 10000 | 4 | MSA & Egyptian
ASTD-B | Twitter | 1600 | 2 | MSA & Egyptian
ArSenTD-Lev | Twitter | 4000 | 5 | Levantine

More details about the aforementioned datasets are found in [the paper.](https://www.aclweb.org/anthology/W19-4608)

In this directory, there are two versions of every dataset:
 1. the original dataset.
 2. the dataset preprocessed using MADAMIRA (with **_final** suffix). This version is used as input to our model.
