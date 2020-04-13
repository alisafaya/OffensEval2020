# SemEval2020

## OffensEval Sub-task A

Offensive Speech Detection

### Macro Averged F1-Scores of test sets

| Model                     | Arabic | Greek | Turkish | Average |
|:-------------------------:|:------:|:-----:|:-------:|:-------:|
| SVM with TF-IDF           | 0.772  | 0.823 | 0.685   | 0.760   |
| Multilingual BERT         | 0.808  | 0.807 | 0.774   | 0.796   |
| Bi-LSTM                   | 0.822  | 0.826 | 0.755   | 0.801   |
| CNN-Text                  | 0.840  | 0.825 | 0.751   | 0.805   |
| BERT <sup>[*](#bert)</sup>| 0.884  | 0.822 | 0.816   | 0.841   |
| BERT-CNN (__Ours__)       | 0.897  | 0.843 | 0.814   | 0.851   |

<a name="bert">*</a>: For language specific pre-trained BERT we use these models: [Arabic](https://github.com/alisafaya/Arabic-BERT), [Greek](https://github.com/nlpaueb/greek-bert), [Turkish](https://github.com/stefan-it/turkish-bert) 