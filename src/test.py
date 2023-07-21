import numpy as np

rng = np.random.default_rng()
for i in range(2):
    random_indexes = rng.choice(100,10,False)
    print(random_indexes)