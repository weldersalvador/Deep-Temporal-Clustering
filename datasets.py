"""
Implementation of the Deep Temporal Clustering model
Dataset loading functions

@author Florent Forest (FlorentF9)
"""

import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from tslearn.datasets import UCR_UEA_datasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from sklearn.preprocessing import LabelEncoder


def load_data(dataset_name=None, n_clusters=3):
    """
    Cria dados sintéticos com padrões distintos para clustering
    """
    n_series = 100  # número de séries por cluster
    timesteps = 48  # comprimento de cada série
    n_features = 1  # número de features por timestep
    
    X_list = []
    y_list = []
    
    # Cluster 0: Ondas senoidais de baixa frequência
    for i in range(n_series):
        t = np.linspace(0, 4*np.pi, timesteps)
        freq = np.random.uniform(0.5, 1.0)
        phase = np.random.uniform(0, 2*np.pi)
        signal = np.sin(freq * t + phase) + np.random.normal(0, 0.1, timesteps)
        X_list.append(signal.reshape(-1, 1))
        y_list.append(0)
    
    # Cluster 1: Ondas senoidais de alta frequência
    for i in range(n_series):
        t = np.linspace(0, 4*np.pi, timesteps)
        freq = np.random.uniform(2.0, 3.0)
        phase = np.random.uniform(0, 2*np.pi)
        signal = np.sin(freq * t + phase) + np.random.normal(0, 0.1, timesteps)
        X_list.append(signal.reshape(-1, 1))
        y_list.append(1)
    
    # Cluster 2: Tendência linear + ruído
    for i in range(n_series):
        slope = np.random.uniform(0.5, 1.5)
        intercept = np.random.uniform(-1, 1)
        t = np.linspace(0, 1, timesteps)
        signal = slope * t + intercept + np.random.normal(0, 0.1, timesteps)
        X_list.append(signal.reshape(-1, 1))
        y_list.append(2)
    
    # Converte para arrays numpy
    X = np.array(X_list)
    y = np.array(y_list)
    
    # Embaralha os dados
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Normaliza os dados
    X_reshaped = X.reshape(-1, timesteps)
    scaler = StandardScaler()
    X_normalized = scaler.fit_transform(X_reshaped)
    X = X_normalized.reshape(-1, timesteps, n_features)
    
    # Divide em treino e teste (80/20)
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    print(f"Usando dataset sintético com padrões:")
    print(f"  - Cluster 0: {np.sum(y_train == 0)} séries senoidais de baixa frequência")
    print(f"  - Cluster 1: {np.sum(y_train == 1)} séries senoidais de alta frequência")
    print(f"  - Cluster 2: {np.sum(y_train == 2)} séries com tendência linear")
    print(f"X_train {X_train.shape}, X_test {X_test.shape}, y_train {y_train.shape}, y_test {y_test.shape}")
    
    return (X_train, y_train), (X_test, y_test)