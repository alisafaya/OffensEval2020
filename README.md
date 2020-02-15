# SemEval2020

## Sub-task A

Offensive Speech Detection

### Set/Model F1-Scores

| Set       | TF-IDF with SVM | ML-DistilBert | Bi-GRU with Attention |
|:---------:|:---------------:|:-------------:|:---------------------:|
| Arabic    | 0.63            | 0.64          | __0.68__              |
| Danish    | 0.46            | 0.42          | __0.47__              |
| Greek     | 0.60            | __0.69__      | 0.60                  |
| Turkish   | 0.40            | __0.54__      | 0.52                  |

### Baseline Scores: (TF-IDF with SVM)

| Set       | Precision | Recall   | F1-Score |
|:---------:|:---------:|:--------:|:--------:|
| Arabic    | 0.66      | 0.60     | 0.63     |
| Danish    | 0.48      | 0.43     | 0.46     |
| Greek     | 0.67      | 0.55     | 0.60     |
| Turkish   | 0.72      | 0.28     | 0.40     |

### ML-DistilBert Scores:

| Set       | Precision | Recall   | F1-Score |
|:---------:|:---------:|:--------:|:--------:|
| Arabic    | 0.68      | 0.60     | 0.64     |
| Danish    | 0.78      | 0.29     | 0.42     |
| Greek     | 0.77      | 0.62     | 0.69     |
| Turkish   | 0.69      | 0.44     | 0.54     |

### Bi-GRU with Attention Scores:

| Set       | Precision | Recall   | F1-Score |
|:---------:|:---------:|:--------:|:--------:|
| Arabic    | 0.74      | 0.63     | 0.68     |
| Danish    | 0.59      | 0.39     | 0.47     |
| Greek     | 0.66      | 0.55     | 0.60     |
| Turkish   | 0.62      | 0.44     | 0.52     |