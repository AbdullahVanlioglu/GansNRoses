import numpy as np
import matplotlib.pyplot as plt 

data1=np.array([-1536,-1259,-979,-64,-63,-68,-64,-61,-59])
data2=np.array([-2125,-1537,-99,-986,-76,-67,-268,-68,-252])
data3=np.array([-2991,-2106,-2105,-1910,-2115,-1918,-1268,-1250,-66])
data4=np.array([-1913,-64,-69,-66,-61,-61,-59])
data5=np.array([-2615,-2118,-1174,-1367,-1173,-1000,-824])


a=1/2941
b=3000/2941


data1=data1*a+b
data2=data2*a+b
data3=data3*a+b
data4=data4*a+b
data5=data5*a+b


mean_data=np.zeros(len(data1))

for i in range(len(data1)):
    if i<7:
        mean_data[i]=(data1[i]+data2[i]+data3[i]+data4[i]+data5[i])/5
    else:
        mean_data[i]=(data1[i]+data2[i]+data3[i])/3

map_num_1=[1,2,3,4,5,6,7,8,9]
map_num_2=[1,2,3,4,5,6,7]

plt.plot(map_num_1,data1,'o:r',color = 'r')
plt.plot(map_num_1,data2,'o:r',color = 'b')
plt.plot(map_num_1,data3,'o:r',color = 'g')
plt.plot(map_num_2,data4,'o:r',color = 'k')
plt.plot(map_num_2,data5,'o:r',color = 'y')
plt.plot(map_num_1,mean_data,linewidth = '5.5')
plt.xlabel("Number of map")
plt.ylabel("Performance")
plt.legend(["Random 1","Random 2","Random 3","Random 4","Random 5","Mean"])
plt.show()
