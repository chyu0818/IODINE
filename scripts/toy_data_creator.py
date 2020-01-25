import numpy as np
import matplotlib.pylab as plt
import gaussian_random_fields as gr

ry = np.arange(0,1,1/1000)
X,Y = np.meshgrid(ry,ry)
for i in range(100):
    X, Y = np.meshgrid(ry, ry)
    ry = np.arange(0, 1, 1/1000)
    Noise = gr.gaussian_random_field(alpha = np.random.uniform(3, 5), size = 1000)
    Noise = Noise + np.abs(np.min(Noise))
    Noise = Noise / np.max(Noise)
    temp = np.random.rand(2, 1)
    min_ind = np.where(ry < min(temp))[-1][-1]
    max_ind = np.where(ry > max(temp))[0][0]
    Signal = np.zeros(Noise.shape) 
    Signal[min_ind:max_ind,min_ind:max_ind] = np.random.rand(1)
    Data = Noise + Signal
    plt.pcolormesh(X,Y,Data,vmin=-2,vmax=2)
    plt.axis('off')
    plt.savefig('/home/bquach/IODINE/gauss_data/train/gauss_{}.png'.format(i), bbox_inches='tight')
