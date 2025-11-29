import matplotlib.pyplot as plt
import numpy as np
from DeepTemporalClustering import DTC
from datasets import load_data

# Carregar dados
(X_train, y_train), (X_test, y_test) = load_data()

# Criar modelo COM OS MESMOS PARÂMETROS DO TREINAMENTO
dtc = DTC(
    n_clusters=3, 
    input_dim=1, 
    timesteps=48, 
    pool_size=8,  # DEVE SER O MESMO DO TREINAMENTO!
    n_filters=50,
    kernel_size=10,
    strides=1,
    n_units=[50, 1],
    alpha=1.0,
    dist_metric='eucl',
    cluster_init='kmeans'
)

# Inicializar modelo
dtc.initialize()

# Carregar pesos
dtc.load_weights('results/tmp/DTC_model_final.weights.h5')

# Predizer clusters
y_pred = dtc.predict(X_test)

print(f"Clusters preditos: {y_pred}")
print(f"Labels verdadeiros: {y_test}")

# Calcular acurácia
from sklearn.metrics import accuracy_score, confusion_matrix
from metrics import cluster_acc

acc = cluster_acc(y_test, y_pred)
print(f"\nAcurácia: {acc:.4f}")

# Visualizar
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
fig.suptitle('Amostras por Cluster Predito', fontsize=16)

for cluster in range(3):
    cluster_samples = X_test[y_pred == cluster][:3]
    for i, sample in enumerate(cluster_samples):
        if len(cluster_samples) > i:
            axes[cluster, i].plot(sample.flatten(), linewidth=2)
            axes[cluster, i].set_title(f'Cluster {cluster} - Amostra {i+1}')
            axes[cluster, i].set_ylabel('Valor Normalizado')
            axes[cluster, i].set_xlabel('Timestep')
            axes[cluster, i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('clusters_visualization.png', dpi=300)
print("\n✅ Visualização salva em 'clusters_visualization.png'")

# Matriz de confusão
from sklearn.metrics import confusion_matrix
import seaborn as sns

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=[f'Cluster {i}' for i in range(3)],
            yticklabels=[f'Real {i}' for i in range(3)])
plt.title('Matriz de Confusão')
plt.ylabel('Cluster Real')
plt.xlabel('Cluster Predito')
plt.savefig('confusion_matrix.png', dpi=300)
print("✅ Matriz de confusão salva em 'confusion_matrix.png'")

plt.show()