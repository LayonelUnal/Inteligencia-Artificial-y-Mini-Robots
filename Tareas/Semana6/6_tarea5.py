from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Crear los datos ---
# Tenemos 10 mensajes cortos — algunos son spam (no deseados) y otros son ham (normales)
docs = ['gratis geld', 'lottery win', 'meeting today', 'buy viagra', 'project update', 'free offer now',
        'important deadline', 'win prize', 'team meeting', 'click here cheap']
labels = ['spam', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam', 'ham', 'spam']

# -- Dividir en entrenamiento y prueba ---
# 60% para entrenar, 40% para probar (el dataset es muy pequeño)
X_train, X_test, y_train, y_test = train_test_split(docs, labels, test_size=0.4, random_state=42)

vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)        # Usa el transformar los datos de prueba

# --- Entrenar el modelo Naive Bayes --
nb = MultinomialNB()  # MultinomialNB es ideal para contar palabras en textos
nb.fit(X_train_vec, y_train)  # El modelo aprende qué palabras son típicas de spam o ham

# Resultos
y_pred = nb.predict(X_test_vec)
print('Accuracy:', accuracy_score(y_test, y_pred))  # Porcentaje de predicciones
print('Preds:', y_pred)