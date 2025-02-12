import librosa
import numpy as np
import joblib

def extract_features(file_path, n_mfcc=13, sr=16000):
    """
    Extrait les MFCCs d'un fichier audio.

    :param file_path: Chemin du fichier audio.
    :param n_mfcc: Nombre de coefficients MFCC à extraire.
    :param sr: Taux d'échantillonnage.
    :return: Tableau des MFCCs moyennés.
    """
    try:
        # Charger le fichier audio
        audio, _ = librosa.load(file_path, sr=sr)
        # Extraire les MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        # Moyenne des MFCCs sur le temps
        return np.mean(mfccs.T, axis=0)
    except Exception as e:
        print(f"Erreur lors de l'extraction des caractéristiques : {e}")
        return None


def test_audio(model_path, file_path):
    """
    Teste un fichier audio avec un modèle SVM.

    :param model_path: Chemin du modèle SVM sauvegardé.
    :param file_path: Chemin du fichier audio à tester.
    """
    # Charger le modèle
    print("Chargement du modèle...")
    model = joblib.load(model_path)

    # Extraire les caractéristiques du fichier audio
    print(f"Extraction des caractéristiques de {file_path}...")
    features = extract_features(file_path)

    if features is None:
        print("Impossible de tester le fichier audio.")
        return

    # Prédire avec le modèle
    print("Prédiction en cours...")
    prediction = model.predict([features])[0]
    confidence = model.predict_proba([features])[0]

    # Afficher les résultats
    if prediction == 1:
        print(f"Mot-clé détecté avec une confiance de {confidence[1] * 100:.2f}%.")
    else:
        print(f"Aucun mot-clé détecté. Confiance : {confidence[0] * 100:.2f}%.")


if __name__ == "__main__":
    # Chemin vers le modèle et le fichier audio
    model_path = "svm_model.pkl"  # Modifiez si nécessaire
    file_path = "my_audio.wav"  # Chemin du fichier audio à tester

    # Tester le fichier audio
    test_audio(model_path, file_path)
