import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

x,y =make_regression(n_samples=200, n_features=1, noise=30)

plt.scatter(x,y)
plt.show()
plt.title('My grafic')