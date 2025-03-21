import numpy as np
import matplotlib.pyplot as plt

show_plots = False


CFL         = 0.5
wave_length = 1
speed       = 1
period      = 1

speed        = wave_length / period
wave_number  = 2 * np.pi / wave_length
angular_freq = speed * wave_number

# Horizontal axis:
Nx = 33
Lx = 1
dx = Lx / (Nx-1)
x = np.arange(0,  Lx + dx , dx)

# Time axis:
dt = CFL * dx / speed
Nt = int(period / dt) + 1
t = np.arange(0, period + dt , dt)

for stencil in ["upwind","downwind","centered"]:  # upwind, downwind, centered

    if stencil == "upwind":
        
        istart = 1
        iend = Nx-1

        stencil_start = -1
        stencil_end   = 0

    elif stencil == "downwind":
        
        istart = 0
        iend = Nx-2

        stencil_start = 0
        stencil_end   = 1

    elif stencil == "centered":
        istart = 1
        iend = Nx-1

        stencil_start = -1
        stencil_end   = 1

    else:
        raise ValueError("Invalid stencil")

    Nstates   = stencil_end - stencil_start + 1
    Ndatasets = (iend-istart) * (Nt-1)*3

    pos = np.zeros((Ndatasets,Nstates))
    features = np.zeros((Ndatasets,Nstates))
    labels   = np.zeros((Ndatasets,1))


    k = 0
    for solution in ["sin","gaussian","shock"]: # sin, shock, gaussian



        for n in range(0, Nt-1): # First and last is the same therefor Nt-1

            time_now  = t[n]
            time_next = t[n+1]
            if solution == "sin":
                u      = np.sin(wave_number * x - angular_freq * time_now )
                u_next = np.sin(wave_number * x - angular_freq * time_next)

            elif solution == "shock":
                
                shock_position_now  = dx/2 + speed * time_now
                shock_position_next = dx/2 + speed * time_next
                
                u =  0*x
                u[x<shock_position_now] = 1 

                u_next =  0*x
                u_next[x<shock_position_next] = 1 

            elif solution == "gaussian":
                sigma         = 0.2
                u      = np.exp(-0.5 * (x - speed * time_now)**2 / sigma**2)
                u_next = np.exp(-0.5 * (x - speed * time_next)**2 / sigma**2)
            else:
                raise ValueError("Invalid solution")

            for i in range(istart,iend): # Upwind discretization therefor starting from 1
                
                a = stencil_start
                b = stencil_end
                if stencil == "centered" and i == iend-1:
                    a = stencil_start - 1
                    b = stencil_end   - 1

                idx = np.arange(i+a,i+b+1)        
                pos[k,:]      = x[idx]
                features[k,:] = u[idx]
                labels[k,0]   = u_next[i]

    

                k = k+1

            if show_plots:
                plt.cla()
                plt.plot(x,u_next,'b')
                plt.plot(x,u_next,'c.')

                plt.plot(x,u,'r')
                plt.plot(x,u,'m.')

#                plt.plot(pos[k,:],features[k,:],'co')
#                plt.plot(x[i],labels[k,0],'mo')
                #plt.legend()
                plt.title(f"Time: {n}")
                plt.grid()
                plt.xlim([0, 1])
                plt.ylim([-1.5, 1.5])
                plt.pause(0.05)  # Pause to update the figure
                plt.draw()  # Redraw the figure
                #input("Tryk på Enter for at fortsætte...")

#        np.savetxt("linear_advection_positions_" + stencil + "_" + solution + ".csv", pos, delimiter=",")
#        np.savetxt("linear_advection_features_" + stencil + "_" + solution + ".csv", features, delimiter=",")
#        np.savetxt("linear_advection_labels_" + stencil + "_" + solution + ".csv", labels, delimiter=",")
    np.savetxt("linear_advection_positions_" + stencil  + ".csv", pos, delimiter=",")
    np.savetxt("linear_advection_features_" + stencil  + ".csv", features, delimiter=",")
    np.savetxt("linear_advection_labels_" + stencil  + ".csv", labels, delimiter=",")

print(f"CFL          : {CFL}")
print(f"wave_length  : {wave_length}")
print(f"period       : {period}")
print(f"speed        : {speed}")
print(f"wave_number  : {wave_number}")
print(f"angular_freq : {angular_freq}")
print(f"dx           : {dx}")
print(f"dt           : {dt}")
print(f"Nx           : {Nx}")
print(f"Nt           : {Nt}")
print(f"Nd           : {Ndatasets}")



#plt.plot(pos.T,features.T)


#plt.plot(pos[:,istart],labels,'b.')
#plt.grid()




plt.show()  # Keep the window open after the loop
