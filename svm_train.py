import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib  # Pour sauvegarder le modèle

def load_audio_files(folder, label, n_mfcc=13, sr=16000):
    """
    Charge les fichiers audio d'un dossier, extrait les MFCCs, et associe une étiquette.
    
    :param folder: Chemin du dossier contenant les fichiers audio.
    :param label: Étiquette associée aux fichiers du dossier.
    :param n_mfcc: Nombre de coefficients MFCC à extraire.
    :param sr: Taux d'échantillonnage.
    :return: Deux listes - une avec les caractéristiques et une avec les étiquettes.
    """
    features = []
    labels = []
    
    for file_name in os.listdir(folder):
        file_path = os.path.join(folder, file_name)
        try:
            # Charger le fichier audio
            audio, _ = librosa.load(file_path, sr=sr)
            # Extraire les MFCCs
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
            # Moyenne des MFCCs sur le temps
            mfccs_mean = np.mean(mfccs.T, axis=0)
            
            features.append(mfccs_mean)
            labels.append(label)
        except Exception as e:
            print(f"Erreur lors du traitement du fichier {file_name}: {e}")
    
    return features, labels


def train_svm_model(keyword_folder, background_folder, model_path="svm_model.pkl"):
    """
    Entraîne un modèle SVM pour détecter un mot-clé à partir des données audio.
    
    :param keyword_folder: Dossier contenant les fichiers du mot-clé.
    :param background_folder: Dossier contenant les fichiers de bruit.
    :param model_path: Chemin pour sauvegarder le modèle SVM.
    """
    # Charger les données
    print("Chargement des fichiers audio...")
    keyword_features, keyword_labels = load_audio_files(keyword_folder, label=1)
    background_features, background_labels = load_audio_files(background_folder, label=0)
    
    # Combiner les données et les étiquettes
    X = np.array(keyword_features + background_features)
    y = np.array(keyword_labels + background_labels)
    
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Créer et entraîner le modèle SVM
    print("Entraînement du modèle SVM...")
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_train, y_train)
    
    # Évaluer le modèle
    print("Évaluation du modèle...")
    y_pred = svm.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(f"Précision: {accuracy_score(y_test, y_pred) * 100:.2f}%")
    
    # Sauvegarder le modèle
    joblib.dump(svm, model_path)
    print(f"Modèle SVM sauvegardé sous : {model_path}")


if __name__ == "__main__":
    # Exemple d'utilisation
    train_svm_model(keyword_folder="keyword_augmented", background_folder="background_augmented")
