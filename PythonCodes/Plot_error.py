import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("Error.txt", delimiter=" ",  skiprows=1, dtype=np.float32)
plt.scatter(np.arange(0,data.shape[0]), data[:,0])
plt.scatter(np.arange(0,data.shape[0]),data[:,1])
plt.xticks(range(data.shape[0]))
plt.legend(("PnP","PnPRansac"))
plt.xlabel('Iteration (K)')
plt.ylabel('Reprojection Error')
plt.show()