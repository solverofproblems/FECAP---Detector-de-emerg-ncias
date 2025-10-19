import os
import glob
import numpy as np
import librosa
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


base_diretorio = "audio_base"

def extrair_mfcc(caminho_arquivo):

    y, sr = librosa.load(caminho_arquivo, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)


arquivos = glob.glob(os.path.join(base_diretorio, "*/*.wav"))
dados = []
rotulos = []

for nome_arquivo in arquivos:
    dados.append(extrair_mfcc(nome_arquivo))
    rotulos.append(os.path.basename(os.path.dirname(nome_arquivo)))


X = np.array(dados)
Y = np.array(rotulos)

print(f'Total de amostras: {len(X)}')

#Separando o treino e o teste:

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)

print(f'Amostras de treino:{len(X_train)}')
print(f'Amostras de teste:{len(X_test)}')

