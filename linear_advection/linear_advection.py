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
u       = u_exact

for n in range(Nt):

    u[1:] = u[1:] + dt * (u[0:-1] - u[1:])/dx

    time    = (n+1) * dt
    u_exact = np.sin(wave_number*x - angular_freq*time)
    u[0]    = u_exact[0]
    

    plt.cla()  # Clear the current plot
    plt.plot(x,u_exact)  # Update y-data
    plt.plot(x,u)  # Update y-data
    plt.title(f"Time: {time:.2f}")
    plt.grid()
    plt.pause(0.05)  # Pause to update the figure
    plt.draw()  # Redraw the figure

plt.show()  # Keep the window open after the loop