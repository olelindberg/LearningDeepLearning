import numpy as np
import matplotlib.pyplot as plt

CFL = 0.5
wave_length = 1
period = 1

speed = wave_length / period
wave_number = 2 * np.pi / wave_length
angular_freq = speed * wave_number

# Horizontal axis:
Nx = 33
dx = wave_length / (Nx-1)
x = np.arange(0, wave_length + dx , dx)

# Time axis:
dt = CFL * dx / speed
Nt = int(period / dt) + 1
t = np.arange(0, period + dt , dt)

Ndatasets = (Nx-2) * (Nt-1)

positions = np.zeros((Ndatasets,2))
features = np.zeros((Ndatasets,2))
labels   = np.zeros((Ndatasets,1))

k = 0
for n in range(0, Nt-1): # First and last is the same therefor Nt-1
    for i in range(1, Nx-1): # Upwind discretization therefor starting from 1

        positions[k,:] = x[i:i+2]
        features[k,:]  = np.sin(wave_number * x[i:i+2] - angular_freq * t[n])
        labels[k,0]    = np.sin(wave_number * x[i]     - angular_freq * t[n+1])

        k = k+1


print(f"CFL          : {CFL}")
print(f"wave_length  : {wave_length}")
print(f"period       : {period}")

print(f"speed        :  {speed}")
print(f"wave_number  : {wave_number}")
print(f"angular_freq : {angular_freq}")

print(f"dx           : {dx}")
print(f"dt           : {dt}")
print(f"Nx           : {Nx}")
print(f"Nt           : {Nt}")

print(f"Nd           : {Ndatasets}")

np.savetxt("linear_advection_features_downwind.csv", features, delimiter=",")
np.savetxt("linear_advection_labels_downwind.csv", labels, delimiter=",")



plt.plot(positions.T,features.T)
plt.plot(positions[:,0],labels,'b.')
plt.grid()

plt.show()  # Keep the window open after the loop
