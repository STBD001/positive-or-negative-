import re
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Funkcja przetwarzania wstępnego
def preprocess_text(text):
    # Konwersja na małe litery
    text = text.lower()
    # Usuwanie zbędnych znaków
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Usuwanie stop-words
    text = ' '.join(word for word in text.split() if word not in ENGLISH_STOP_WORDS)
    return text

# Przykładowe dane
text = [
    "I love this movie, it's fantastic!", # Pozytywny
    "This film was a waste of time, really bad.", # Negatywny
    "Absolutely amazing, I had a great time watching it.", # Pozytywny
    "The best movie I have ever seen, brilliant production.", # Pozytywny
    "Terrible plot, I don't recommend this movie.", # Negatywny
    "Tragic film, script, set design and acting." # Negatywny
]

labels = [1, 0, 1, 1, 0, 0]

# Przetwarzanie wstępne
text = [preprocess_text(t) for t in text]

# Podział danych na zestaw treningowy i testowy
X_train, X_test, Y_train, Y_test = train_test_split(text, labels, test_size=0.25, random_state=42)

# Inicjalizacja TF-IDF wektoryzatora
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Trenowanie modelu
model = LogisticRegression()
model.fit(X_train_vec, Y_train)

# Predykcja i ocena dokładności
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(Y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Funkcja do przewidywania sentymentu
def predict_sentiment(text_input):
    text_input = preprocess_text(text_input)
    text_vec = vectorizer.transform([text_input])
    prediction = model.predict(text_vec)
    return "Pozytywny" if prediction[0] == 1 else "Negatywny"

# Interaktywne wprowadzenie tekstu
if __name__ == "__main__":
    user_input = input("Wprowadź tekst do oceny sentymentu: ")
    sentiment = predict_sentiment(user_input)
    print(f"Sentiment: {sentiment}")
