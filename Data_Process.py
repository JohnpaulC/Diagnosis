from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np

import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)

dataset = np.load('satellite_state.npy')
X = dataset[:, :-1]
y = dataset[-1]
x_train, x_test, y_train, y_test = train_test_split(
    dataset[:, :-1], dataset[:, -1])

scaler = StandardScaler().fit(x_train)
#x_train = scaler.transform(x_train)

pca = PCA(n_components=0.98, whiten=True)
pca.fit(x_train)

X_new = pca.transform(x_train)
fig1 = plt.figure(1, figsize=(12, 8))

colors = ['b', 'r', 'orange']
Label_Com = ['positive', 'negative']

for index in range(2):
    x_1 = X_new[y_train == index][:, 0]
    x_2 = X_new[y_train == index][:, 1]

    plt.scatter(x_1, x_2, c=colors[index], cmap='brg', alpha=0.2, marker='o', linewidth=0)
plt.xlabel("First Principal Component", fontsize=20)
plt.ylabel("Second Principal Component", fontsize=20)

plt.savefig("PCA.png")
plt.show()


print(pca.explained_variance_ratio_)
arg_sort = np.absolute(pca.components_).argsort()
print(arg_sort)
print(pca.components_)

plt.figure(figsize=(12, 8))
for i in range(2):
    plt.subplot(2, 1, i + 1)
    component = np.absolute(pca.components_[i])
    softmax = np.exp(component) / sum(np.exp(component))
    plt.bar(range(45), softmax)
    plt.xlabel("Feature Index", fontsize=15)
    plt.ylabel("Softmax Ratio", fontsize=15)
plt.savefig("PCA_matrix.png")
plt.show()
