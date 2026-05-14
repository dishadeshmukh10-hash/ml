import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from catboost import CatBoostClassifier

x,y=make_classification(n_samples=200,n_features=2,n_redundant=0,n_clusters_per_class=1,random_state=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

model=CatBoostClassifier(verbose=0)   #verbose controls:how much training information is printed on screen,here do not print training logs

model.fit(x_train,y_train)

y_pred=model.predict(x_test)

print("Accuracy\n",accuracy_score(y_test,y_pred))
print("Confusion Matrix\n",confusion_matrix(y_test,y_pred))

x1=np.linspace(x[:,0].min()-1,x[:,0].max()+1,200)
x2=np.linspace(x[:,1].min()-1,x[:,1].max()+1,200)

xx,yy=np.meshgrid(x1,x2)

Z=model.predict(np.c_[xx.ravel(),yy.ravel()])
Z=Z.reshape(xx.shape)

plt.figure()
plt.contourf(xx,yy,Z,alpha=0.3)
plt.scatter(x[:,0],x[:,1],c=y)
plt.title("CatBoost")
plt.show(block=True)