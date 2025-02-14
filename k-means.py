from sklearn.datasets import make_blobs  # Importa a função para gerar dados fictícios
import matplotlib.pyplot as plt        # Importa a biblioteca para criação de gráficos
import seaborn as sns                 # Importa a biblioteca Seaborn (opcional para estilos)
from sklearn.cluster import KMeans    # Importa o algoritmo de clusterização K-Means

# Gerar dados fictícios com 300 amostras, 4 centros e um desvio padrão de 0.60
x, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# Treinar o modelo K-Means com 4 clusters
# n_init=10 é importante para versões mais recentes do scikit-learn
kmeans = KMeans(n_clusters=4, random_state=0, n_init=10)
y_kmeans = kmeans.fit_predict(x)  # Ajusta o modelo e obtém os rótulos dos clusters

# Plotar os pontos de dados com cores baseadas nos clusters identificados
plt.scatter(x[:, 0], x[:, 1], c=y_kmeans, s=50, cmap='viridis')

# Obter os centros dos clusters após o treinamento
target_centers = kmeans.cluster_centers_

# Plotar os centros dos clusters em preto com marcadores grandes ('X')
plt.scatter(target_centers[:, 0], target_centers[:, 1], c='black', s=200, alpha=0.5, marker='X')

# Adicionar título e rótulos aos eixos
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

# Exibir o gráfico
plt.show()
