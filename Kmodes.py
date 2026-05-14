import numpy as np
from kmodes.kmodes import KModes

data = np.array([
    ['Red', 'Small'],
    ['Blue', 'Large'],
    ['Red', 'Medium'],
    ['Blue', 'Small'],
    ['Red', 'Large']
])

model = KModes(n_clusters=2, random_state=1)

labels = model.fit_predict(data)

print("Cluster Labels:\n", labels)