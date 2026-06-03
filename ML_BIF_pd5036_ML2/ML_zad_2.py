# Importowanie bibliotek
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# Tworzymy przykładowy zbiór danych
data = {
'TP53_expr': [2.1, 8.5, 1.8, 6.2, 7.9, 3.1, 9.2, 2.8],
'BRCA1_expr': [3.4, 7.2, 2.5, 6.1, 6.8, 4.0, 7.9, 3.9],
'TF_motifs': [2, 6, 1, 4, 5, 2, 6, 3],
'KRAS': [1.2, 7.1, 0.9, 6.8, 1.5, 5.5, 1.0, 6.3],
'Cancer_status': [0, 1, 0, 1, 1, 0, 1, 0]
} 
df = pd.DataFrame(data)
# Cechy (X) i etykiety (y)
X = df[['TP53_expr', 'BRCA1_expr', 'TF_motifs', 'KRAS']]
y = df['Cancer_status']
# Podział danych
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25,
random_state=42, stratify=y)
#Stworzenie modelu
model = LogisticRegression()
#Trenowanie modelu
model.fit(X_train, y_train)
#Predykcja
y_pred = model.predict(X_test)
#Ocena modelu
print("Dokładność:", accuracy_score(y_test, y_pred))        
print("Raport klasyfikacji:\n", classification_report(y_test, y_pred))

# Ocena modelu:
# Accuracy = 1.00
# Precision = 1.00
# Recall = 1.00
# F1-score = 1.00
#
# Model poprawnie sklasyfikował wszystkie próbki testowe.
# Macierz konfuzji nie zawiera błędnych klasyfikacji.
# Należy jednak pamiętać, że zbiór testowy zawierał tylko 2 próbki,
# dlatego wynik 100% nie gwarantuje równie dobrej skuteczności
# na większych i bardziej zróżnicowanych danych.
