# Importation des bibliothèques nécessaires
import pandas as pd
import numpy as np
import nltk
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix

# Télécharger les ressources NLTK nécessaires
nltk.download('stopwords')
from nltk.corpus import stopwords

# Chargement du dataset IMDb Reviews (assurez-vous d'avoir téléchargé le dataset depuis Kaggle)
# Remplacez 'path_to_your_dataset.csv' par le chemin du fichier .csv sur votre machine
try:
    df = pd.read_csv('IMDB.csv')
    print("Fichier chargé avec succès !")
    print(df.head())  # Vérifie le contenu du fichier
except Exception as e:
    print(f"Erreur lors du chargement du fichier : {e}")
    exit()

# Vérification de la présence des colonnes nécessaires
if 'review' not in df.columns or 'sentiment' not in df.columns:
    print("Colonnes 'review' ou 'sentiment' manquantes dans le dataset.")
    exit()

# Exploration des classes de sentiment
try:
    sns.countplot(x='sentiment', data=df)
    plt.title("Distribution des sentiments")
    plt.show()
except Exception as e:
    print(f"Erreur lors de l'affichage de la distribution des sentiments : {e}")
    exit()

# Prétraitement des textes
def preprocess_text(text):
    # Convertir en minuscules
    text = text.lower()
    # Supprimer les chiffres, les symboles et les ponctuations
    text = ''.join([char for char in text if char.isalpha() or char.isspace()])
    # Tokenisation et suppression des stop words
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)

# Appliquer le prétraitement sur les critiques
print("Prétraitement des critiques...")
df['cleaned_review'] = df['review'].apply(preprocess_text)
print("Critiques prétraitées avec succès !")

# Vérifier les premières lignes après prétraitement
print(df[['review', 'cleaned_review']].head())

# Séparer les données en features (X) et labels (y)
X = df['cleaned_review']
y = df['sentiment']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorisation des textes (compte de mots)
vectorizer = CountVectorizer(max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Entraînement du modèle Naive Bayes
model = MultinomialNB()
model.fit(X_train_vec, y_train)
print("Modèle entraîné avec succès !")

# Prédictions sur les données de test
y_pred = model.predict(X_test_vec)

# Évaluation du modèle
print("Classification Report :\n", classification_report(y_test, y_pred))
print("Confusion Matrix :")
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.title('Matrice de confusion')
plt.show()

# Fonction pour prédire le sentiment d'une nouvelle critique
def predict_sentiment(review, vectorizer, model):
    # Prétraiter le texte (enlever stopwords, mettre en minuscules, etc.)
    review = preprocess_text(review)
    print(f"Critique prétraitée : {review}")  # Vérifie le texte prétraité
    
    # Vectoriser la critique prétraitée
    review_vec = vectorizer.transform([review])
    print(f"Vecteur de la critique : {review_vec.shape}")  # Vérifie la forme du vecteur
    
    # Prédire le sentiment avec le modèle
    prediction = model.predict(review_vec)
    
    # Retourner la prédiction (Positive ou Negative)
    return prediction[0]

# Exemple d'utilisation avec une nouvelle critique
new_review = "I loved this movie! It was amazing and the storyline was very touching."
prediction = predict_sentiment(new_review, vectorizer, model)
print(f"Sentiment de la critique : {prediction}")
