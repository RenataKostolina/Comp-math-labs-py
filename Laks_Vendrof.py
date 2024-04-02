import numpy as np
import matplotlib.pyplot as plt 
import pylab as plb
import math
from PIL import Image

#Начальные значения 
T = 18.0
L = 20.0
h = 0.5
CFL = [0.6, 1.0, 1.01]
tau = [CFL[0]*h, CFL[1]*h, CFL[2]*h]

x = np.array([(i-1)*h for i in range(1,int(L/h + 1) + 1)])
u_n1 = np.sin(4*np.pi*x/L)

#Какой Курант?
n = 0

#Лакс-Вендрофф
t = 0.0

k=0
while t < T:
    u_n = u_n1.copy()

    for i in range(1,len(x)-1):
        C = CFL[n]
        u_n1[i]= (C*C + C)/2*u_n[i-1] + (1-C*C)*u_n[i] + (C*C-C)/2*u_n[i+1]
        u_n1[len(x)-1]= (C*C + C)/2*u_n[len(x)-2] + (1-C*C)*u_n[len(x)-1] + (C*C-C)/2*u_n[0]
        u_n1[0]= u_n1[len(x)-1]

    t += tau[n]
    plb.plot(x, u_n1,'purple')
    plb.title(r'CFL = '+ str(CFL[n])) 
    plb.xlim([0,20])
    plb.ylim([-1.2, 1.2])
    plb.grid()
    plb.savefig('image'+ str(n) + str(k)+'.jpg')
    plb.close()
    k += 1

arr = np.arange(1,k)
def save2Gif(rang):
    image = Image.open("image" + str(n) + "0.jpg")
    images=[]
    for i in rang:
        if i!=0:
            name = 'image'+ str(n) + str(i)+'.jpg'
            images.append(Image.open(name))
    image.save('LaksV'+str(n)+'.gif', save_all=True, append_images=images,loop=100,duration=1)
save2Gif(arr)

print("done!")