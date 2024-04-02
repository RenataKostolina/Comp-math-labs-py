import math
import numpy as np
import matplotlib.pyplot as plt
import pylab as plb
from PIL import Image

#Начальные значения 
L = 10.
T = 0.04
gamma = 5./3.

v_l = 20.
rho_l = 6.
p_l = 460.894
e_l = p_l/((gamma-1)*rho_l)

v_r = -6.2
rho_r = 6.
p_r = 46.095
e_r = p_r/((gamma-1)*rho_r)

time = np.linspace(0, T, 201)
tau = T/(len(time)-1)

coordinates = np.linspace(-L, L, 201)
h = 2*L/(len(coordinates)-1)

#Создаем массив решений
w = np.zeros([len(time)+1, len(coordinates), 3])
w[0] = np.array([np.array([rho_l, rho_l*v_l, rho_l*e_l]) for x in coordinates[:100]] 
              + [np.array([rho_r, rho_r*v_r, rho_r*e_r]) for x in coordinates[100:]])


for n in range(0, len(time)):
    for l in range(0, len(coordinates)):
        u = w[n][l][1]/w[n][l][0]
        e = w[n][l][2]/w[n][l][0]
        c = math.sqrt(gamma*(gamma-1)*e)

        omega = np.array([[-u*c, c, gamma-1], 
                          [-c*c, 0, gamma-1], 
                          [u*c, -c, gamma-1]])
        
        omega_inv = np.array([[1/(2*c*c),       -1/(c*c),          1/(2*c*c)], 
                              [(u+c)/(2*c*c),   -1*u/(c*c),    (u-c)/(2*c*c)], 
                              [1/(2*(gamma-1)),  0,          1/(2*(gamma-1))]])
        
        lambdabs = np.diag([abs(u+c), abs(u), abs(u-c)])

        A = np.array([[0,            1,              0], 
                      [-u*u,         2*u,      gamma-1], 
                      [-gamma*u*e,   gamma*e,        u]])
        
        if l == len(coordinates)-1:
            w[n+1][l] = w[n+1][l-1]
        elif l == 0:
            w[n+1][l] = w[n][l+1] - tau*(A @ (w[n][l+2]-w[n][l])/(2*h)) + tau*(omega_inv @ lambdabs @ omega) @ (w[n][l+2]-2*w[n][l+1]+w[n][l])/(2*h)
        else:
            w[n+1][l] = w[n][l] - tau*(A @ (w[n][l+1]-w[n][l-1])/(2*h)) + tau*(omega_inv @ lambdabs @ omega) @ (w[n][l+1]-2*w[n][l]+w[n][l-1])/(2*h)
        
        CFL = max(abs(u+c), abs(u), abs(u-c))*tau/h
        if (CFL > 1): print(":(")
     
    rho = w[n, :, 0]
    """ """
    plb.plot(coordinates,rho,'pink')
    plb.xlabel(r'$x, м$')
    plb.ylabel(r'$\rho, кг/м^3$')
    plb.xlim([-10,10])
    plb.ylim([0,20])
    plb.grid()
    plb.savefig('imagerho'+str(n)+'.jpg')
    plb.close()
    """  
    
    u = w[n, :, 1]/rho 
    plb.plot(coordinates,u,'plum')
    plb.xlabel(r'$x, м$')
    plb.ylabel(r'$U, м/c$')
    plb.xlim([-10,10])
    plb.ylim([-50,250])
    plb.grid()
    plb.savefig('imageu'+str(n)+'.jpg')
    plb.close()
    
    e = w[n, :, 2]/rho
    e /= 1.e3
    plb.plot(coordinates,e,'skyblue')
    plb.xlabel(r'$x, м$')
    plb.ylabel(r'$e, кДж/кг$')
    plb.xlim([-10,10])
    plb.ylim([60,180])
    plb.grid()
    plb.savefig('imagee'+str(n)+'.jpg')
    plb.close()
    e = w[n, :, 2]/rho
    p = (gamma-1)*rho*e
    p /= 1.e2
    plb.plot(coordinates,p,'yellowgreen')
    plb.xlabel(r'$x, м$')
    plb.ylabel(r'$p, атм$')
    plb.xlim([-10,10])
    plb.ylim([0,12000])
    plb.grid()
    plb.savefig('imagep'+str(n)+'.jpg')
    plb.close()
    """
    
print("done")