import sounddevice as sd
import wave

def record_audio(file_name="output.wav", duration=3, samplerate=16000):
    """
    Enregistre de l'audio depuis le microphone et sauvegarde dans un fichier WAV.

    :param file_name: Nom du fichier de sortie (par défaut : output.wav)
    :param duration: Durée de l'enregistrement en secondes
    :param samplerate: Taux d'échantillonnage en Hz (par défaut : 16000)
    """
    print(f"Enregistrement en cours pour {duration} secondes...")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Attendre la fin de l'enregistrement
    print("Enregistrement terminé !")

    # Sauvegarder les données audio dans un fichier WAV
    with wave.open(file_name, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16 bits = 2 octets
        wf.setframerate(samplerate)
        wf.writeframes(audio_data.tobytes())
    print(f"Fichier audio sauvegardé sous : {file_name}")

# Exemple d'utilisation
if __name__ == "__main__":
    record_audio(file_name="my_audio.wav", duration=3)
