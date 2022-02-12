import numpy as np
import matplotlib.pyplot as plt 

data=np.loadtxt("Random_rewards.txt")

data_gan=np.loadtxt("gan_experimental_rewards.txt")

a=1/2941
b=3000/2941

random_var = []
gan_var = []

data=data*a+b
data_gan=data_gan*a+b

mean_data=np.zeros(len(data[0,:]))

mean_data_gan=np.zeros(len(data_gan[0:,]))

for i in range(len(data[0,:])):
    #sum_col=0
    random_var.append(np.var(data[:,i]))
    for j in range(len(data[:,0])):
        mean_data[i]+=data[j,i]/len(data[:,0])

for i in range(len(data_gan[0,:])):
    #sum_col=0
    gan_var.append(np.var(data_gan[:,i]))
    for j in range(len(data_gan[:,0])):
        mean_data_gan[i]+=data_gan[j,i]/len(data_gan[:,0])

"""for i in range(len(data_gan)):
mean_data_gan[i]+=data_gan[i]/len(data_gan)"""


map_num=[1,2,3,4,5,6,7,8,9,10]

# for i in range(len(data[:,0])):
#     plt.plot(map_num,data[i,:],ls=':',color='r')

# for i in range(len(data_gan[:,0])):
#     plt.plot(map_num,data_gan[i,:],ls=':',color='b')

"""for i in range(len(data_gan)):
plt.plot(map_num,data_gan[i],ls=':',color='b')"""

# plt.plot(map_num,mean_data,linewidth = '3.5',label="Mean_Random",color='r')
# plt.plot(map_num,mean_data_gan,linewidth = '3.5',label="Mean_Gan",color='b')
# plt.xlabel("Number of maps")
# plt.ylabel("Performance")
# plt.show() 



# print("gan_var: ",gan_var)
# print("random_var: ",random_var)

plt.plot(map_num, gan_var, linewidth = '3.5',label="GAN VAR",color='b')
plt.plot(map_num, random_var, linewidth = '3.5',label="Random VAR",color='r')
plt.legend(loc='upper right')
plt.show()