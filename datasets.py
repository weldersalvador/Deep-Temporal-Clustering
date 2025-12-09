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

ucr = UCR_UEA_datasets()
all_ucr_datasets = ucr.list_datasets()


def load_data(dataset_name='CBF', n_clusters=3,n_samples_per_cluster=100, 
              timesteps=48, noise_level=0.05, test_split=0.2, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Verificar se é dataset UCR
    if dataset_name in all_ucr_datasets:
        print(f"Carregando dataset UCR: {dataset_name}")
        
        # Dividir em treino e teste
        X_train_ucr, y_train_ucr, X_test_ucr, y_test_ucr = ucr.load_dataset(dataset_name)
        X_scaled = np.concatenate((X_train_ucr, X_test_ucr))
        y = np.concatenate((y_train_ucr, y_test_ucr))
        
        split = int((1 - test_split) * len(X_scaled))
        X_train, X_test = X_scaled[:split], X_scaled[split:]
        y_train, y_test = y[:split], y[split:]
        
        print(f"Dataset: {dataset_name}")
        print(f"X_train {X_train.shape}, X_test {X_test.shape}")
        print(f"y_train {y_train.shape}, y_test {y_test.shape}")
        print(f"Número de clusters: {len(np.unique(y_train))}")
        
        return (X_train, y_train), (X_test, y_test)