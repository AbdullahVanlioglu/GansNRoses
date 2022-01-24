import numpy as np
import matplotlib.pyplot as plt 

data=np.loadtxt("Random_rewards.txt")

data_gan=np.loadtxt("gan_experimental_rewards.txt")

a=1/2941
b=3000/2941


data=data*a+b
data_gan=data_gan*a+b

mean_data=np.zeros(len(data[0,:]))

mean_data_gan=np.zeros(len(data_gan))

for i in range(len(data[0,:])):
    #sum_col=0
    for j in range(len(data[:,0])):
        mean_data[i]+=data[j,i]/len(data[:,0])

"""for i in range(len(data_gan)):
mean_data_gan[i]+=data_gan[i]/len(data_gan)"""


map_num=[1,2,3,4,5,6,7,8,9,10]

for i in range(len(data[:,0])):
    plt.plot(map_num,data[i,:],ls=':',color='r')

"""for i in range(len(data_gan)):
plt.plot(map_num,data_gan[i],ls=':',color='b')"""

plt.plot(map_num,mean_data,linewidth = '3.5',label="Mean_Random",color='r')
plt.plot(map_num,data_gan,linewidth = '3.5',label="Mean_Gan",color='b')
plt.xlabel("Number of maps")
plt.ylabel("Performance")
plt.show() 