import os
import glob
import numpy as np
import librosa
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import joblib
import warnings
warnings.filterwarnings('ignore')

class AudioClassifierSimples:
    def __init__(self, base_diretorio="audio_base"):
        self.base_diretorio = base_diretorio
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        self.history = None
        
    def extrair_mfcc_melhorado(self, caminho_arquivo):
        """
        Extrai MFCCs com configurações otimizadas para classificação de fala
        """
        try:
            # Carregar áudio com sample rate padronizado
            y, sr = librosa.load(caminho_arquivo, sr=22050)
            
            # Normalizar o áudio
            y = librosa.util.normalize(y)
            
            # Extrair MFCCs com parâmetros otimizados
            mfcc = librosa.feature.mfcc(
                y=y, 
                sr=sr, 
                n_mfcc=13,  # 13 coeficientes MFCC
                n_fft=2048,  # Tamanho da janela FFT
                hop_length=512,  # Overlap entre janelas
                n_mels=128  # Número de filtros mel
            )
            
            # Calcular estatísticas dos MFCCs
            mfcc_mean = np.mean(mfcc.T, axis=0)
            mfcc_std = np.std(mfcc.T, axis=0)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta_mean = np.mean(mfcc_delta.T, axis=0)
            
            # Adicionar características espectrais básicas
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            
            # Combinar todas as features
            features = np.concatenate([
                mfcc_mean,           # 13 features
                mfcc_std,            # 13 features  
                mfcc_delta_mean,     # 13 features
                [spectral_centroid], # 1 feature
                [zero_crossing_rate] # 1 feature
            ])
            
            return features
            
        except Exception as e:
            print(f"Erro ao processar {caminho_arquivo}: {e}")
            return np.zeros(42)  # Retorna array de zeros com tamanho fixo
    
    def augment_audio_simples(self, y, sr):
        """
        Data augmentation simples mas eficaz
        """
        augmented = []
        
        # 1. Mudança de velocidade (time stretching)
        y_fast = librosa.effects.time_stretch(y, rate=1.2)
        y_slow = librosa.effects.time_stretch(y, rate=0.8)
        augmented.extend([y_fast, y_slow])
        
        # 2. Mudança de pitch
        y_pitch_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=1)
        y_pitch_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-1)
        augmented.extend([y_pitch_up, y_pitch_down])
        
        # 3. Adição de ruído leve
        noise_factor = 0.01
        y_noise = y + noise_factor * np.random.randn(len(y))
        augmented.append(y_noise)
        
        return augmented
    
    def carregar_dados(self, aplicar_augmentacao=True):
        """
        Carrega dados com augmentation opcional
        """
        arquivos = glob.glob(os.path.join(self.base_diretorio, "*/*.wav"))
        dados = []
        rotulos = []
        
        print(f"Encontrados {len(arquivos)} arquivos de áudio")
        
        for nome_arquivo in arquivos:
            try:
                # Extrair features do arquivo original
                features = self.extrair_mfcc_melhorado(nome_arquivo)
                if len(features) > 0:
                    dados.append(features)
                    rotulos.append(os.path.basename(os.path.dirname(nome_arquivo)))
                
                # Aplicar data augmentation se solicitado
                if aplicar_augmentacao:
                    y, sr = librosa.load(nome_arquivo, sr=22050)
                    y = librosa.util.normalize(y)
                    augmented_samples = self.augment_audio_simples(y, sr)
                    
                    for aug_sample in augmented_samples:
                        features_aug = self.extrair_mfcc_melhorado_audio(aug_sample, sr)
                        if len(features_aug) > 0:
                            dados.append(features_aug)
                            rotulos.append(os.path.basename(os.path.dirname(nome_arquivo)))
                            
            except Exception as e:
                print(f"Erro ao processar {nome_arquivo}: {e}")
                continue
        
        return np.array(dados), np.array(rotulos)
    
    def extrair_mfcc_melhorado_audio(self, y, sr):
        """
        Versão da extração de MFCCs que recebe o áudio diretamente
        """
        try:
            # Normalizar o áudio
            y = librosa.util.normalize(y)
            
            # Extrair MFCCs
            mfcc = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=13, n_fft=2048, 
                hop_length=512, n_mels=128
            )
            
            # Calcular estatísticas
            mfcc_mean = np.mean(mfcc.T, axis=0)
            mfcc_std = np.std(mfcc.T, axis=0)
            mfcc_delta = librosa.feature.delta(mfcc)
            mfcc_delta_mean = np.mean(mfcc_delta.T, axis=0)
            
            # Características espectrais
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            zero_crossing_rate = np.mean(librosa.feature.zero_crossing_rate(y))
            
            # Combinar features
            features = np.concatenate([
                mfcc_mean, mfcc_std, mfcc_delta_mean,
                [spectral_centroid], [zero_crossing_rate]
            ])
            
            return features
            
        except Exception as e:
            return np.zeros(42)
    
    def criar_modelo_simples(self, input_dim, num_classes):
        """
        Modelo neural simples mas eficaz
        """
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_dim=input_dim),
            layers.Dropout(0.3),
            
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            
            layers.Dense(16, activation='relu'),
            layers.Dropout(0.1),
            
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def treinar_modelo(self, X, y, test_size=0.2):
        """
        Treina o modelo com validação
        """
        # Normalizar os dados
        X_scaled = self.scaler.fit_transform(X)
        
        # Codificar labels
        y_encoded = self.encoder.fit_transform(y)
        num_classes = len(np.unique(y_encoded))
        
        print(f"Total de amostras: {len(X_scaled)}")
        print(f"Número de classes: {num_classes}")
        print(f"Dimensão das features: {X_scaled.shape[1]}")
        
        # Dividir em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=test_size, 
            random_state=42, stratify=y_encoded
        )
        
        print(f"Amostras de treino: {len(X_train)}")
        print(f"Amostras de teste: {len(X_test)}")
        
        # Criar modelo
        model = self.criar_modelo_simples(X_scaled.shape[1], num_classes)
        
        # Callbacks para melhorar o treinamento
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=0.0001
        )
        
        # Treinar modelo
        print("\nIniciando treinamento...")
        self.history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=100,
            batch_size=16,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Avaliar modelo
        print("\n=== AVALIAÇÃO DO MODELO ===")
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Acurácia no conjunto de teste: {test_acc:.4f}")
        
        # Fazer predições
        y_pred = model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Relatório de classificação
        print("\n=== RELATÓRIO DE CLASSIFICAÇÃO ===")
        print(classification_report(y_test, y_pred_classes, 
                                  target_names=self.encoder.classes_))
        
        # Plotar curva de aprendizado
        self.plot_learning_curves()
        
        return model, X_test, y_test, y_pred_classes

    
    def plot_learning_curves(self):
        """
        Plota as curvas de aprendizado (accuracy)
        """
        if self.history is None:
            print("Nenhum histórico de treinamento disponível.")
            return
        
        plt.figure(figsize=(10, 6))
        
        # Curva de Accuracy
        plt.plot(self.history.history['accuracy'], label='Treino', linewidth=2)
        plt.plot(self.history.history['val_accuracy'], label='Validação', linewidth=2)
        plt.title('Curva de Acurácia', fontsize=14, fontweight='bold')
        plt.xlabel('Épocas')
        plt.ylabel('Acurácia')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Análise do treinamento
        print("\n=== ANÁLISE DO TREINAMENTO ===")
        final_train_acc = self.history.history['accuracy'][-1]
        final_val_acc = self.history.history['val_accuracy'][-1]
        best_val_acc = max(self.history.history['val_accuracy'])
        best_epoch = self.history.history['val_accuracy'].index(best_val_acc) + 1
        
        print(f"Acurácia final de treino: {final_train_acc:.4f}")
        print(f"Acurácia final de validação: {final_val_acc:.4f}")
        print(f"Melhor acurácia de validação: {best_val_acc:.4f} (época {best_epoch})")
        

def main():
    """
    Função principal para executar o treinamento
    """
    print("=== CLASSIFICADOR DE ÁUDIO SIMPLES E EFICAZ ===")
    print("Foco em MFCCs e técnicas essenciais para máxima acurácia\n")
    
    # Inicializar classificador
    classifier = AudioClassifierSimples()
    
    # Carregar dados
    print("Carregando e processando dados...")
    X, y = classifier.carregar_dados(aplicar_augmentacao=True)
    
    # Treinar modelo
    model, X_test, y_test, y_pred = classifier.treinar_modelo(X, y)
    
    model.save('modelo_finalizado.h5')
    joblib.dump(classifier.scaler, 'scaler.pkl')
    print('Modelo salvo com sucesso!')
    print('Scaler salvo em scaler.pkl')

if __name__ == "__main__":
    main()
