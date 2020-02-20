# SemEval2020

## Sub-task A

Offensive Speech Detection

### Set/Model 4 Folds F1-Scores

| Set       | TF-IDF with SVM | ML-DistilBert | Bi-GRU with Attention | Word-CNN model        |
|:---------:|:---------------:|:-------------:|:---------------------:|:---------------------:|
| Arabic    | 0.63            | 0.64          | 0.68                  | __0.70__              |
| Danish    | 0.46            | 0.42          | 0.47                  | __0.51__              |
| Greek     | 0.60            | __0.69__      | 0.60                  | 0.61                  |
| Turkish   | 0.40            | __0.54__      | 0.52                  | 0.49                  |

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

### Word-CNN Scores:

| Set       | Precision | Recall   | F1-Score |
|:---------:|:---------:|:--------:|:--------:|
| Arabic    | 0.85      | 0.59     | 0.70     |
| Danish    | 0.75      | 0.39     | 0.51     |
| Greek     | 0.69      | 0.55     | 0.61     |
| Turkish   | 0.62      | 0.40     | 0.49     |

### Language Specific Models

| Set                 | Precision | Recall   | F1-Score |
|:-------------------:|:---------:|:--------:|:--------:|
| Turkish (Bert)      | 0.76      | 0.63     | 0.69     |
| Greek   (Bert-CNN)  | 0.71      | 0.72     | 0.71     |
| Danish  (Bert-CNN)  | 0.67      | 0.33     | 0.44     |
| Turkish (Bert-CNN)  | 0.65      | 0.64     | 0.64     |
| Greek   (Bert-LSTM) | 0.81      | 0.63     | 0.71     |
| Danish  (Bert-LSTM) | 0.61      | 0.35     | 0.45     |
| Turkish (Bert-LSTM) | 0.60      | 0.67     | 0.64     |
| Greek   (Bert-Attn) | 0.73      | 0.70     | 0.72     |