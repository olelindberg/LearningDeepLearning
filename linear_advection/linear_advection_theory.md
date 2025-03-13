
$$
y = w_1 x_1 + w_2 x_2 - b
$$

$$
\begin{align}
F &= \frac{1}{2} y^2 \nonumber \\ 
  &= (w_1 x_1 + w_2 x_2 - b)*(w_1 x_1 + w_2 x_2 - b)  \nonumber \\
  &= w_1^2 x_1^2 + 2 w_1 w_2 x_1 x_2 + w_2^2 x_2^2 - 2 w_1 x_1 b - 2 w_2 x_2 b + b^2
\end{align}
$$

Den partielle afledte med hensyn til  $w_1$ :

$$
\frac{\partial f}{\partial w_1} = 2 w_1 x_1^2 + 2 w_2 x_1 x_2 - 2 x_1 b
$$

Den partielle afledte med hensyn til $w_2$:

$$
\frac{\partial f}{\partial w_2} = 2 w_1 x_1 x_2 + 2 w_2 x_2^2 - 2 x_2 b
$$


Jacobian vector:

$$
\mathbf{J} = \begin{bmatrix} 
2 w_1 x_1^2 + 2 w_2 x_1 x_2 - 2 x_1 b \\ 
2 w_2 x_2^2 + 2 w_1 x_1 x_2 - 2 x_2 b 
\end{bmatrix}
$$

Hessian matrix:

$$
\mathbf{H} = \begin{bmatrix}
2 x_1^2 + 2 x_1 x_2 & 2 x_1 x_2           \\
2 x_1 x_2           & 2 x_2^2 + 2 x_1 x_2
\end{bmatrix}
$$