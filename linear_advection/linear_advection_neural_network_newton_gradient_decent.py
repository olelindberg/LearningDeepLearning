import numpy as np
import matplotlib.pyplot as plt

def residual(x1,x2,w1,w2,b):
    return x1*w1 + x2*w2 - b

def jacobian(x1,x2,w1,w2,b):
    J = [2*w1*x1**2 + 2*w2*x1*x2 - 2*x1*b, 2*w2*x2**2 + 2*w1*x1*x2 - 2*x2*b ]
    return J

def newton_method(lhs,rhs, epochmax=10, tol=1e-6):

    w1 = 0
    w2 = 0

    rmse = []

    for epoch in range(epochmax):

        loss = []

        for i in range(lhs.shape[0]):

            x1 = lhs[i,0]
            x2 = lhs[i,1]
            b  = rhs[i]

            if x1!=0 and x2!=0:

                J = jacobian(x1,x2,w1,w2,b)
                
                H = [[2*x1**2 + 2*x1*x2, 2*x1*x2         ],          
                    [2*x1*x2         , 2*x2**2 + 2*x1*x2]]

                dx = np.linalg.solve(H,J)

                w1 = w1 - dx[0]
                w2 = w2 - dx[1]

            loss.append((residual(x1,x2,w1,w2,b)))

        rmse.append(np.sqrt(np.mean(np.array(loss)**2)))
        if rmse[-1] < tol:
            break

                    
    return w1,w2,rmse

def gradient_decent(lhs,rhs, epochmax=10, lr=0.01, tol=1e-6):

    w1 = 0
    w2 = 0

    rmse = []

    for epoch in range(epochmax):

        loss = []

        for i in range(lhs.shape[0]):

            x1 = lhs[i,0]
            x2 = lhs[i,1]
            b  = rhs[i]

            if x1!=0 and x2!=0:

                J = jacobian(x1,x2,w1,w2,b)
                
                w1 = w1 - lr*J[0]
                w2 = w2 - lr*J[1]


            loss.append((residual(x1,x2,w1,w2,b)))

        rmse.append(np.sqrt(np.mean(np.array(loss)**2)))
        if rmse[-1] < tol:
            break               

    return w1,w2,rmse



lhs = np.loadtxt('linear_advection_features.csv', delimiter=',')
rhs = np.loadtxt('linear_advection_labels.csv',   delimiter=',')

epochmax = 1000
lr = 0.6
tol = 1e-10
gradient_decent_lr = [0.1,0.2,0.3,0.4,0.5]
w1,w2,mse_newton = newton_method(lhs,rhs,epochmax=epochmax,tol=tol)
print('Newton Method')
print(w1,w2)

err_gradient_decent = []
for lr in gradient_decent_lr:
    print('Gradient Decent ' + str(lr))
    w1,w2,err = gradient_decent(lhs,rhs,epochmax=epochmax,lr=lr,tol=tol)
    err_gradient_decent.append(err)
    print(w1,w2)

plt.figure()
plt.plot(np.log10(mse_newton),label='Newton Method')

for i in range(len(err_gradient_decent)):
    plt.plot(np.log10(err_gradient_decent[i]),label='Gradient Decent lr = ' + str(gradient_decent_lr[i]))

plt.legend()
plt.grid()
plt.xlabel('Epoch')
plt.ylabel('log10(rmse)')
plt.title('Newton Method vs Gradient Decent')
plt.ylim(-16, 0)

plt.show()

