#Zaimportowanie biblioteki pandas
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

#Wczytanie przykładowego pliku CSV (dostępnego tutaj)
df = pd.read_csv(r"C:\Users\defaultuser0.DESKTOP-9B20QHN\Desktop\PJATK\Uczenie maszynowe w bioinformatyce\ML_BIF_pd5036\ML_BIF_pd5036_projetkt1\dane_projekt1.csv")
#Wyświetlenie podstawowych statystyk
print(df.head())
#print(df.info())
#print(df["Gene_Function"].value_counts())
print(df.describe())
X = df.drop(columns=["Gene_Function", "Gene_ID"])
y = df["Gene_Function"]
print(X.head())
print(y.head())
# Dzielimy dane na 80% treningowe + walidacyjne, 20% testowe
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
# Dzielimy dane treningowe na 75% trening + 25% walidacja (czyli 60%/20% globalnie)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, stratify=y_temp, random_state=42)
# Sprawdzenie sortowania danych
print("Rozmiar zbioru treningowego:", X_train.shape)
print("Rozmiar zbioru walidacyjnego:", X_val.shape)
print("Rozmiar zbioru testowego:", X_test.shape)        
print("Rozmiar zbioru treningowego:", y_train.shape)
print("Rozmiar zbioru walidacyjnego:", y_val.shape)
print("Rozmiar zbioru testowego:", y_test.shape)  
# Sprawdzenie rozkładu klas w zbiorach
stratify=y_train    
print("Rozkład klas w zbiorze treningowym:\n", y_train.value_counts())
#print("Rozkład klas w zbiorze walidacyjnym:\n", y_val.value_counts())
print("Rozkład klas w zbiorze testowym:\n", y_test.value_counts())
# Inicjalizacja modelu Random Forest    
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "SVM": SVC(kernel="rbf"),
    "KNN": KNeighborsClassifier(n_neighbors=3)
}
for name, model in models.items():
    model.fit(X_train, y_train)
    print(f"\nTrenowanie modelu: {name}")
# Predykcje na zbiorze testowym
    y_pred = model.predict(X_test)  
# Ocena modelu na zbiorze walidacyjnym
#val_score = model.score(X_val, y_val)
#print("Dokładność na zbiorze walidacyjnym:", val_score)
# Ocena modelu na zbiorze testowym
#test_score = model.score(X_test, y_test)
#print("Dokładność na zbiorze testowym:", test_score)

#print("Accuracy:", accuracy_score(y_test, y_pred))
#print(classification_report(y_test, y_pred))
    print("\n", name)
    print("Accuracy:", accuracy_score(y_test, y_pred))


#WNIOSKI:
#Celem projektu było zbudowanie modelu uczenia maszynowego przewidującego funkcję genów na podstawie danych ekspresji RNA-Seq. 
#Dane zostały podzielone na zbiór treningowy (60%), walidacyjny (20%) oraz testowy (20%), 
# a następnie porównano skuteczność trzech algorytmów klasyfikacji: Random Forest, SVM oraz KNN.
#
#Najlepsze wyniki uzyskał model SVM (Support Vector Machine), osiągając dokładność klasyfikacji równą 40%. 
# Modele Random Forest oraz KNN uzyskały dokładność 20%, co oznacza, że klasyfikowały poprawnie jedynie około 2 z 10 próbek testowych.
#
#Otrzymane wyniki wskazują, że zastosowane modele miały trudności z poprawnym rozpoznawaniem funkcji genów. 
# Najbardziej prawdopodobną przyczyną jest niewielka liczba dostępnych danych (50 genów), podział na cztery klasy oraz syntetyczny charakter zbioru danych. 
# Przy tak małej liczbie próbek modele nie są w stanie nauczyć się wystarczająco dobrze zależności pomiędzy poziomem ekspresji genów a ich funkcją biologiczną.
#Spośród porównywanych algorytmów najlepsze rezultaty osiągnął model SVM, dlatego można uznać go za najbardziej odpowiedni dla analizowanego zbioru danych. 
# W celu poprawy skuteczności klasyfikacji w przyszłości na pewno warto zwiększyć liczbę próbek treningowych lub przeprowadzić optymalizację parametrów modeli.