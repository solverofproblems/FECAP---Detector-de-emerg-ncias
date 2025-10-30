
import speech_recognition as sr
import pyaudio as py
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import joblib


r = sr.Recognizer()

with sr.Microphone() as source:
    print('Estamos te escutando...')
    audio = r.listen(source)

with open('audio_capturado.wav', "wb") as audio_salvado:
    audio_salvado.write(audio.get_wav_data())

"""
Extrai exatamente as mesmas 42 features usadas no treinamento em modelo_completo.py:
13 MFCC mean + 13 MFCC std + 13 Delta MFCC mean + 1 spectral centroid + 1 ZCR
"""
y, sr = librosa.load('audio_capturado.wav', sr=22050)
y = librosa.util.normalize(y)

mfcc = librosa.feature.mfcc(
    y=y,
    sr=sr,
    n_mfcc=13,
    n_fft=2048,
    hop_length=512,
    n_mels=128
)

mfcc_mean = np.mean(mfcc.T, axis=0)
mfcc_std = np.std(mfcc.T, axis=0)
mfcc_delta = librosa.feature.delta(mfcc)
mfcc_delta_mean = np.mean(mfcc_delta.T, axis=0)

spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
zero_cross = np.mean(librosa.feature.zero_crossing_rate(y))

X_novo = np.concatenate([
    mfcc_mean,
    mfcc_std,
    mfcc_delta_mean,
    [spectral_centroid],
    [zero_cross]
]).reshape(1, -1)

# Aplicar o mesmo scaler usado no treinamento
scaler = joblib.load('scaler.pkl')
X_novo = scaler.transform(X_novo)

modelo_treinado = load_model('modelo_finalizado.h5')
predicao = modelo_treinado.predict(X_novo)

labels = ["ambulancia", "bombeiro", "policia"]  # substitua pelos seus nomes reais
indice = np.argmax(predicao)  # pega o índice da maior probabilidade

print("Classe prevista:", labels[indice])


