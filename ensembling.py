import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

cnn = np.load("word_cnn_probs1.npy")
bi_gru = np.load("bi_gru_probs.npy")
bi_gru_emd = np.load("bi_gru_probs_emb.npy")

y_test = np.load("word_cnn_gold.npy").tolist()

avg_probs = (cnn + bi_gru) / 2.0
avg_probs_emb = (cnn + bi_gru_emd) / 2.0

y_pred = [1 if s >= 0.5 else 0 for s in avg_probs]
y_pred_emd = [1 if s >= 0.5 else 0 for s in avg_probs_emb]

cnn = [1 if s >= 0.5 else 0 for s in cnn]
bi_gru = [1 if s >= 0.5 else 0 for s in bi_gru]
bi_gru_emd = [1 if s >= 0.5 else 0 for s in bi_gru_emd]

print(classification_report(y_test, y_pred))
print(classification_report(y_test, y_pred_emd))

# print(classification_report(y_test, cnn))
print(classification_report(y_test, bi_gru))
print(classification_report(y_test, bi_gru_emd))
