import os
import glob
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt


base_diretorio = "audio_base"

def extrair_mfcc(caminho_arquivo):

    y, sr = librosa.load(caminho_arquivo, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfcc.T, axis=0)


def aumentando_arquivos(y, sr):
    y_pitch = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
    y_stretch = librosa.effects.time_stretch(y, rate=1.1)
    return [y, y_pitch, y_stretch]


arquivos = glob.glob(os.path.join(base_diretorio, "*/*.wav"))
dados = []
rotulos = []

for nome_arquivo in arquivos:

    y, sr = librosa.load(nome_arquivo, sr=None)
    audio_aumentado = aumentando_arquivos(y, sr)
    for y_aumentado in audio_aumentado:
        mfcc = np.mean(librosa.feature.mfcc(y=y_aumentado, sr=sr, n_mfcc=13).T, axis=0)
        dados.append(mfcc)
        rotulos.append(os.path.basename(os.path.dirname(nome_arquivo)))


X = np.array(dados)
Y = np.array(rotulos)

encoder = LabelEncoder()
Y_encoded = encoder.fit_transform(Y)


print(f'Total de amostras: {len(X)}')

#Separando o treino e o teste:

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_encoded, test_size=0.2, random_state=42, stratify=Y_encoded
)

print(f'Amostras de treino:{len(X_train)}')
print(f'Amostras de teste:{len(X_test)}')


num_classes = len(np.unique(Y_encoded))


model = keras.Sequential([
    layers.Dense(32, activation='relu', input_dim=13),
    layers.Dropout(0.3),
    layers.Dense(16, activation='relu'),
    layers.Dropout(0.3), 
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Treinando o modelo:

history = model.fit(X_train, Y_train, epochs=400, verbose=0)
print('Treinamento concluído.')

#Avaliando no conjunto de teste:

loss, acc = model.evaluate(X_test, Y_test, verbose=0)
print(f"A acurácia do conjunto de teste é de: {acc:.2f}")

# Previsão no teste:

plt.plot(history.history['accuracy'])
plt.title('Evolução da Acurácia no Treinamento')
plt.xlabel('Épocas')
plt.ylabel("Acurácia")
plt.grid(True)
plt.show()