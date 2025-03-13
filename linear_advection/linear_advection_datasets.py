import numpy as np
import matplotlib.pyplot as plt

CFL = 0.5
wave_length = 1
period = 1

speed = wave_length / period
wave_number = 2 * np.pi / wave_length
angular_freq = speed * wave_number

# Horizontal axis:
Nx = 100
dx = wave_length / Nx
x = np.arange(0, 2 * dx, dx)

# Time axis:
dt = CFL * dx / speed
Nt = int(period / dt) + 1
t = np.arange(0, period + dt, dt)

x, t = np.meshgrid(x, t)


u_features = np.zeros((Nt, 2))
u_features[:, 0:2] = np.sin(wave_number * x - angular_freq * t)
# u_features[:,2] = speed
# u_features[:,0] = np.float32(np.random.random([Nt-1]))
# u_features[:,1] = np.float32(np.random.random([Nt-1]))
u_labels = np.sin(wave_number * x[:, 1] - angular_freq * (t[:, 1] + dt))

print(t.shape)
print(x.shape)
print(u_features.shape)
print(u_labels.shape)

print("wave_number  " + str(wave_number))
print("angular_freq " + str(angular_freq))
print("period       " + str(period))


np.savetxt("linear_advection_features.csv", u_features, delimiter=",")
np.savetxt("linear_advection_labels.csv", u_labels, delimiter=",")


plt.plot(np.transpose(x), np.transpose(u_features[:, 0:2]))
plt.plot(x[:, 1], u_labels, "o")
plt.grid()

plt.figure()
plt.plot(t[:, 1], u_labels)
plt.grid()

plt.show()  # Keep the window open after the loop
