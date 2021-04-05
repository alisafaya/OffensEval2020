# OffensEval2020

```
@inproceedings{safaya-etal-2020-kuisail,
    title = "{KUISAIL} at {S}em{E}val-2020 Task 12: {BERT}-{CNN} for Offensive Speech Identification in Social Media",
    author = "Safaya, Ali  and
      Abdullatif, Moutasem  and
      Yuret, Deniz",
    booktitle = "Proceedings of the Fourteenth Workshop on Semantic Evaluation",
    month = dec,
    year = "2020",
    address = "Barcelona (online)",
    publisher = "International Committee for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2020.semeval-1.271",
    pages = "2054--2059",
}
```

## OffensEval Sub-task A

Offensive Speech Detection: paper [here](https://www.aclweb.org/anthology/2020.semeval-1.271/)

### Macro Averaged F1-Scores of test sets

| Model                     | Arabic | Greek | Turkish | Average |
|:-------------------------:|:------:|:-----:|:-------:|:-------:|
| SVM with TF-IDF           | 0.772  | 0.823 | 0.685   | 0.760   |
| Multilingual BERT         | 0.808  | 0.807 | 0.774   | 0.796   |
| Bi-LSTM                   | 0.822  | 0.826 | 0.755   | 0.801   |
| CNN-Text                  | 0.840  | 0.825 | 0.751   | 0.805   |
| BERT <sup>[*](#bert)</sup>| 0.884  | 0.822 | __0.816__   | 0.841   |
| BERT-CNN (__Ours__)       | __0.897__  | __0.843__ | 0.814   | __0.851__   |

<a name="bert">*</a>: For language specific pre-trained BERT we use these models: [Arabic](https://github.com/alisafaya/Arabic-BERT), [Greek](https://github.com/nlpaueb/greek-bert), [Turkish](https://github.com/stefan-it/turkish-bert) 
