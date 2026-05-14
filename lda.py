import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

x,y = make_classification(n_samples=200,n_clusters_per_class=1,n_features=2,n_redundant=0,random_state=1)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=1)

model = LinearDiscriminantAnalysis()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("Accuracy :",accuracy_score(y_test,y_pred))
print("Confusion Matrix :\n",confusion_matrix(y_test,y_pred))

plt.scatter(x[:,0],x[:,1],c=y)
plt.title("LDA")
plt.show()