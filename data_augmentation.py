import os
import librosa
import numpy as np
from scipy.io.wavfile import write

def augment_audio(file_path, output_folder, sr=16000):
    """
    Applique des augmentations sur un fichier audio et sauvegarde les résultats.

    :param file_path: Chemin du fichier audio d'entrée.
    :param output_folder: Dossier où sauvegarder les fichiers augmentés.
    :param sr: Taux d'échantillonnage.
    """
    try:
        # Charger le fichier audio
        audio, _ = librosa.load(file_path, sr=sr)

        # Appliquer des augmentations
        # 1. Ajout de bruit
        noisy_audio = audio + 0.01 * np.random.normal(0, 1, len(audio))

        # 2. Pitch shifting
        pitch_shifted_up = librosa.effects.pitch_shift(audio, sr=sr, n_steps=2)
        pitch_shifted_down = librosa.effects.pitch_shift(audio, sr=sr, n_steps=-2)

        # 3. Time stretching
        stretched_slow = librosa.effects.time_stretch(audio, rate=0.8)
        stretched_fast = librosa.effects.time_stretch(audio, rate=1.2)

        # Préparer le dossier de sortie
        os.makedirs(output_folder, exist_ok=True)

        # Sauvegarder les fichiers augmentés
        base_name = os.path.basename(file_path).split('.')[0]
        write(os.path.join(output_folder, f"{base_name}_noisy.wav"), sr, (noisy_audio * 32767).astype(np.int16))
        write(os.path.join(output_folder, f"{base_name}_pitch_up.wav"), sr, (pitch_shifted_up * 32767).astype(np.int16))
        write(os.path.join(output_folder, f"{base_name}_pitch_down.wav"), sr, (pitch_shifted_down * 32767).astype(np.int16))
        write(os.path.join(output_folder, f"{base_name}_stretched_slow.wav"), sr, (stretched_slow * 32767).astype(np.int16))
        write(os.path.join(output_folder, f"{base_name}_stretched_fast.wav"), sr, (stretched_fast * 32767).astype(np.int16))

        print(f"Augmentations effectuées pour {file_path}.")

    except Exception as e:
        print(f"Erreur lors de l'augmentation de {file_path}: {e}")


def augment_dataset(input_folder, output_folder, sr=16000):
    """
    Applique des augmentations à tous les fichiers audio dans un dossier.

    :param input_folder: Dossier contenant les fichiers audio originaux.
    :param output_folder: Dossier où sauvegarder les fichiers augmentés.
    :param sr: Taux d'échantillonnage.
    """
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.wav'):
            file_path = os.path.join(input_folder, file_name)
            augment_audio(file_path, output_folder, sr=sr)


if __name__ == "__main__":
    # Dossier contenant les fichiers audio originaux
    input_folder = "background"  # Changez le chemin selon vos besoins

    # Dossier pour sauvegarder les fichiers augmentés
    output_folder = "background_augmented"

    # Appliquer les augmentations
    augment_dataset(input_folder, output_folder)
