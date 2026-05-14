import matplotlib
matplotlib.use('TkAgg')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

x=np.random.rand(100, 1)*10
y=4*x[:,0]+3+np.random.randn(100)

print("X Values\n",x[:10])
print("Y values\n",y[:10])

model = LinearRegression()
model.fit(x,y)

pred = model.predict(x)

print("Slope = ",model.coef_[0])
print("Intercept = ",model.intercept_)
print("MSE = ",mean_squared_error(y,pred))
print("R2 Score = ",r2_score(y,pred))

plt.scatter(x,y)
plt.plot(x,pred)
plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.show(block=True)
