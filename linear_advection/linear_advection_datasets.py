import numpy as np
import matplotlib.pyplot as plt

advection_speed = 1

# Horizontal axis:
Lx = 1
Nx = 100
dx = Lx / Nx
x  = np.arange(0,2*dx,dx)

# Time axis:
CFL = 0.5
dt  = CFL * dx / advection_speed
T   = 1
Nt  = int(T/dt)+1
t   = np.arange(0,T+dt,dt)

x,t = np.meshgrid(x,t)

wave_number  = 2 * np.pi / Lx
angular_freq = wave_number * advection_speed

u_features =np.zeros((Nt,2))
u_features[:,0:2] = 0.5*np.sin(wave_number*x      - angular_freq*t)+0.5
#u_features[:,2] = advection_speed
#u_features[:,0] = np.float32(np.random.random([Nt-1]))
#u_features[:,1] = np.float32(np.random.random([Nt-1]))
u_labels   = 0.5*np.sin(wave_number*x[:,1] - angular_freq*(t[:,1]+dt))+0.5

print(t)
print(x)
print(u_features)
print(u_labels)

plt.plot(np.transpose(x), np.transpose(u_features[:,0:2]))
plt.plot(x[:,1],u_labels,'o')
plt.grid()

np.savetxt('linear_advection_features.csv', u_features, delimiter=',')
np.savetxt('linear_advection_labels.csv',   u_labels,   delimiter=',')

plt.show()  # Keep the window open after the loop