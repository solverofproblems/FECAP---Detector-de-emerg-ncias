import os
import glob
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

def extrair_mfcc_simples(caminho_arquivo):
    """
    Extrai apenas MFCCs básicos - muito simples
    """
    try:
        y, sr = librosa.load(caminho_arquivo, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfcc.T, axis=0)  # Apenas a média dos MFCCs
    except:
        return np.zeros(13)

def augment_audio_basico(y, sr):
    """
    Data augmentation muito simples
    """
    # Mudança de velocidade
    y_fast = librosa.effects.time_stretch(y, rate=1.1)
    y_slow = librosa.effects.time_stretch(y, rate=0.9)
    
    return [y_fast, y_slow]

def carregar_dados_simples(base_diretorio="audio_base"):
    """
    Carrega dados de forma bem simples
    """
    arquivos = glob.glob(os.path.join(base_diretorio, "*/*.wav"))
    dados = []
    rotulos = []
    
    print(f"Encontrados {len(arquivos)} arquivos")
    
    for arquivo in arquivos:
        # Arquivo original
        mfcc = extrair_mfcc_simples(arquivo)
        dados.append(mfcc)
        rotulos.append(os.path.basename(os.path.dirname(arquivo)))
        
        # Data augmentation simples
        try:
            y, sr = librosa.load(arquivo, sr=22050)
            augmented = augment_audio_basico(y, sr)
            
            for aug_y in augmented:
                mfcc_aug = librosa.feature.mfcc(y=aug_y, sr=sr, n_mfcc=13)
                dados.append(np.mean(mfcc_aug.T, axis=0))
                rotulos.append(os.path.basename(os.path.dirname(arquivo)))
        except:
            continue
    
    return np.array(dados), np.array(rotulos)

def criar_modelo_trivial(input_dim, num_classes):
    """
    Modelo neural muito simples
    """
    model = keras.Sequential([
        keras.layers.Dense(32, activation='relu', input_dim=input_dim),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    """
    Função principal - bem simples
    """
    print("=== CLASSIFICADOR TRIVIAL DE ÁUDIO ===")
    print("Apenas MFCCs + rede neural simples\n")
    
    # Carregar dados
    print("Carregando dados...")
    X, y = carregar_dados_simples()
    
    print(f"Total de amostras: {len(X)}")
    print(f"Features por amostra: {X.shape[1]}")
    
    # Codificar labels
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    num_classes = len(np.unique(y_encoded))
    
    # Dividir dados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Treino: {len(X_train)} amostras")
    print(f"Teste: {len(X_test)} amostras")
    
    # Criar e treinar modelo
    model = criar_modelo_trivial(X.shape[1], num_classes)
    
    print("\nTreinando modelo...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=16,
        verbose=1
    )
    
    # Avaliar
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nAcurácia no teste: {test_acc:.3f}")
    
    # Plotar curva simples
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['accuracy'], label='Treino')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title('Curva de Acurácia')
    plt.xlabel('Épocas')
    plt.ylabel('Acurácia')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Salvar modelo
    model.save('modelo_trivial.h5')
    print("Modelo salvo como 'modelo_trivial.h5'")

if __name__ == "__main__":
    main()
