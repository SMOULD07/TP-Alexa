from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
import joblib

def load_audio_files(folder, label, n_mfcc=13, sr=16000):
    features = []
    labels = []
    
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        try:
            audio, _ = librosa.load(file_path, sr=sr)
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
            mfccs_mean = np.mean(mfccs.T, axis=0)
            features.append(mfccs_mean)
            labels.append(label)
        except Exception as e:
            print(f"Erreur lors du traitement du fichier {file_name}: {e}")
    
    return features, labels


def train_knn_model(keyword_folder, background_folder, model_path="knn_model.pkl", n_neighbors=5):
    # Charger les données
    print("Chargement des fichiers audio...")
    keyword_features, keyword_labels = load_audio_files(keyword_folder, label=1)
    background_features, background_labels = load_audio_files(background_folder, label=0)
    
    X = np.array(keyword_features + background_features)
    y = np.array(keyword_labels + background_labels)
    
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Créer et entraîner le modèle KNN
    print("Entraînement du modèle KNN...")
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)
    
    # Évaluer le modèle
    print("Évaluation du modèle...")
    y_pred = knn.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(f"Précision: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    
    # Sauvegarder le modèle
    joblib.dump(knn, model_path)
    print(f"Modèle KNN sauvegardé sous : {model_path}")


if __name__ == "__main__":
    train_knn_model(keyword_folder="keyword_augmented", background_folder="background_augmented")
