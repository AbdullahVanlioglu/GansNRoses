import numpy as np
import matplotlib.pyplot as plt 

data=np.loadtxt("Performance_3.txt")
data_gan=np.loadtxt("Performance-gan_all.txt")
mean=np.zeros(10)
mean_gan=np.zeros(10)
map_num=np.linspace(1,10,10)

for i in range(10):
    mean[i]=np.mean(data[:,i])
    mean_gan[i]=np.mean(data_gan[:,i])

for i in range(len(data)):
    plt.plot(map_num,data[i,:],linestyle = 'dotted',c='r')

for i in range(len(data_gan)):
    plt.plot(map_num,data_gan[i,:],linestyle = 'dotted',c='b')

plt.plot(map_num,mean, linewidth = '3.5',c='r')
plt.plot(map_num,mean_gan, linewidth = '3.5',c='b')
plt.show()

