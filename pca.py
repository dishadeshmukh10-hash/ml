import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA

data=load_iris()
x=data.data
y=data.target

model=PCA(n_components=2)  #reduce data to 2 dimensions
x_pca=model.fit_transform(x)

plt.figure()
plt.scatter(x_pca[:,0],x_pca[:,1],c=y)
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA")
plt.show()