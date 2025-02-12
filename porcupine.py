import pvporcupine
import sounddevice as sd
import numpy as np
import time

def detect_keyword():
    # Remplacez VOTRE_ACCESS_KEY par la clé obtenue sur Picovoice
    access_key = "wXwznj5Gi/r4loIv0GWevt19r26tKT0PaSdrpLJjqx3KfGk2ReRssg=="

    # Initialiser Porcupine pour détecter "Alexa"
    porcupine = pvporcupine.create(access_key=access_key, keywords=["alexa"])
    
    # Taille de frame attendue par Porcupine
    frame_length = porcupine.frame_length

    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        # Vérifier que la taille des frames est correcte
        if len(indata) != frame_length:
            print(f"Taille des frames incorrecte : attendu {frame_length}, reçu {len(indata)}")
            return

        pcm_data = (indata[:, 0] * 32767).astype(np.int16)  # Convertir en PCM 16 bits
        keyword_index = porcupine.process(pcm_data)
        if keyword_index >= 0:
            print("Mot-clé détecté : Alexa !")

    # Configurer l'entrée audio
    with sd.InputStream(
        callback=audio_callback,
        channels=1,
        samplerate=porcupine.sample_rate,
        blocksize=frame_length,
        dtype="float32"
    ):
        print("Écoute en cours... Appuyez sur Ctrl+C pour arrêter.")
        try:
            while True:
                time.sleep(0.1)  # Laisser du temps au processeur pour traiter
        except KeyboardInterrupt:
            print("Arrêt manuel détecté. Fermeture de l'application.")

# Lancer la détection
if __name__ == "__main__":
    detect_keyword()
