import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.decomposition import TruncatedSVD

data=load_iris()
x=data.data
y=data.target

model=TruncatedSVD(n_components=2)
X_SVD=model.fit_transform(x)

plt.figure()
plt.scatter(X_SVD[:,0],X_SVD[:,1],c=y)
plt.xlabel("SVD Component 1")
plt.ylabel("SVD Component 2")

plt.title("SVD")

plt.show()