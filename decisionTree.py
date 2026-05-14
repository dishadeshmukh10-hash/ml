import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import accuracy_score,confusion_matrix

x,y=make_classification(n_samples=200, n_clusters_per_class=1, n_redundant=0, n_features=2, random_state=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)

model=DecisionTreeClassifier(max_depth=3)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

print("Accuracy = ",accuracy_score(y_test,y_pred))
print("Confusion Matrix = ",confusion_matrix(y_test,y_pred))

plt.figure(figsize=(8,5))
plt.title("Decision Trees")
plot_tree(model,filled=True)
plt.show(block=True)
