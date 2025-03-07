import numpy as np
import matplotlib.pyplot as plt

advection_speed = 1

# Horizontal axis:
Lx = 1
Nx = 100
dx = Lx / Nx
x  = np.arange(0,Lx,dx)

# Time axis:
CFL = 0.5
dt = CFL * dx / advection_speed
Nt = 100

print(f"CFL: {CFL}")

wave_number  = 2 * np.pi / Lx
angular_freq = wave_number * advection_speed

u_exact = np.sin(wave_number*x)
u       = np.array(u_exact)
u_ls    = np.array(u_exact)
u_nn    = np.array(u_exact)

for n in range(Nt):

    u[1:]    = u[1:] + dt * (u[0:-1] - u[1:])/dx
    u_ls[1:] = 0.50010616*u_ls[0:-1] + 0.50005777*u_ls[1:]
    u_nn[1:] = 0.3123*u_nn[0:-1] + 0.6966*u_nn[1:]
    
 
    time    = (n+1) * dt
    u_exact = np.sin(wave_number*x - angular_freq*time)
    u[0]    = u_exact[0]
    u_ls[0] = u_exact[0]
    u_nn[0] = u_exact[0]

    plt.cla()
    plt.plot(x,u_exact)
    plt.plot(x,u)
    plt.plot(x,u_ls) 
    plt.plot(x,u_nn)
    plt.title(f"Time: {time:.2f}")
    plt.grid()
    plt.ylim([-1, 1])
    plt.pause(0.05)  # Pause to update the figure
    plt.draw()  # Redraw the figure

plt.show()  # Keep the window open after the loop