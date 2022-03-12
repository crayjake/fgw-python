import numpy as np
import math

# Variables
js = np.array(list(range(1, 1001)))
D_t = 10000
D = 50000
h = 25000
N = 0.02
rho_s = 1

# Space
Z = np.linspace(0, D, 10000)

# Normalisation coefficient
# Halliday+ has (this value)^(-1) ...?
A = math.sqrt(2 / (rho_s * (N ** 2) * D))

# Z dependence
def phi(_z, j):
    z = np.copy(_z)
    return np.exp(z / (2 * h)) * np.sin((j * np.pi * z) / D) * A

# True value for vertical form of heating
def trueZ(z):
    return np.sin(np.pi * z / D_t) * np.exp(z / (2 * h)) * (np.heaviside(z, 1) - np.heaviside(z - D_t, 1))



# Loop through modes and increment heating by S_j
S = np.empty(Z.shape)

for i in range(len(js)):
    j = js[i]

    # special case (in our case when j = 5)
    if ((j * D_t / D) - 1) == 0:
        S_j = A * (rho_s / 2) * D_t
    else:
        S_A = np.sin((np.pi) * ((j * D_t / D) - 1)) / ((j * D_t / D) - 1)
        S_B = np.sin((np.pi) * ((j * D_t / D) + 1)) / ((j * D_t / D) + 1)
        S_j = A * (rho_s / 2) * (D_t / np.pi) * (S_A - S_B)

    S += ((N ** 2) * S_j * phi(Z, j))



# Plot the results and true value
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 150


fig, ax = plt.subplots()
ax.plot(Z, S, f'k') # Black dashed for Fourier series


ax.plot(Z, trueZ(Z), f'r:') # Red dotted for true value

plt.title(f'Heating Fourier n={len(js)}')
plt.show()