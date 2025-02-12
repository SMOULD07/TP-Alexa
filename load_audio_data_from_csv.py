import sounddevice as sd
import wave
import os

def record_audio(file_name, duration=3, samplerate=16000):
    """
    Enregistre de l'audio depuis le microphone et sauvegarde dans un fichier WAV.

    :param file_name: Nom du fichier de sortie (inclut le chemin et le nom).
    :param duration: Durée de l'enregistrement en secondes (par défaut : 2s).
    :param samplerate: Taux d'échantillonnage en Hz (par défaut : 16000).
    """
    print(f"Enregistrement en cours pour {duration} secondes...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Attendre la fin de l'enregistrement
    print("Enregistrement terminé !")

    # Sauvegarder les données audio dans un fichier WAV
    os.makedirs(os.path.dirname(file_name), exist_ok=True)  # Créer le dossier si nécessaire
    with wave.open(file_name, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16 bits = 2 octets
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())
    print(f"Fichier audio sauvegardé sous : {file_name}")


def batch_record(keyword_folder="keyword", background_folder="background", num_keyword=10, num_background=10):
    """
    Enregistre plusieurs fichiers audio pour le mot-clé et le bruit de fond.

    :param keyword_folder: Dossier pour enregistrer les fichiers "Alexa".
    :param background_folder: Dossier pour enregistrer les fichiers bruit.
    :param num_keyword: Nombre de fichiers pour le mot-clé.
    :param num_background: Nombre de fichiers pour le bruit.
    """
    print("Commencez par enregistrer les fichiers contenant le mot-clé (par ex : 'Alexa').")
    for i in range(num_keyword):
        file_name = f"{keyword_folder}/alexa_{i + 1}.wav"
        print(f"Préparation pour l'enregistrement du fichier : {file_name}")
        input("Appuyez sur Entrée pour commencer l'enregistrement...")
        record_audio(file_name, duration=3)

    print("\nPassez maintenant aux fichiers de bruit (par ex : 'Bruit de fond').")
    for i in range(num_background):
        file_name = f"{background_folder}/noise_{i + 1}.wav"
        print(f"Préparation pour l'enregistrement du fichier : {file_name}")
        input("Appuyez sur Entrée pour commencer l'enregistrement...")
        record_audio(file_name, duration=3)


# Exemple d'utilisation
if __name__ == "__main__":
    batch_record(num_keyword=10, num_background=10)  # Enregistrer 5 fichiers pour chaque classe
