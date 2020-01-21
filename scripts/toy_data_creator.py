import numpy as np
import matplotlib.pylab as plt

ry = np.arange(0,1,1/1000)
X,Y = np.meshgrid(ry,ry)
Noise = np.random.rand(1000,1000)
min_ind = np.where(ry < 0.3)[-1][-1]
max_ind = np.where(ry > 0.6)[0][0]
Signal = np.zeros(Noise.shape) 
Signal[min_ind:max_ind,min_ind:max_ind] = np.random.rand(1)
Data = Noise + Signal
plt.pcolormesh(X,Y,Data,vmin=0,vmax=2)
