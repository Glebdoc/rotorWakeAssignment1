#Code for Lifting Line group assignment 
# import necessary libraries
import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd
import scipy.linalg as la

#import data from csv file
data = pd.read_excel('polars.xlsx')
#import the cl and cd data
polar_cl = data['Cl'].to_numpy()
polar_cd = data['Cd'].to_numpy()
polar_alpha = data['Alfa'].to_numpy()

#define geometry of wind turbine
Uinf = 10 # unperturbed wind speed in m/s
NBlades = 3
TipLocation_R =  1
RootLocation_R =  0.2
Radius = 50 # meters
pitch = 2 # degrees

#define the radial discretization
N = 100  # number of radial elements
r = np.linspace(RootLocation_R,TipLocation_R,N) # radius in meters

#define simple simple case of rectangular wing with constant chord to test the code and constant Cl
def Cl(alpha):
    return 2 * np.pi * np.sin(alpha)
alpha = 5 * np.pi / 180  # Angle of attack in radians
AR = 25
S = 1.0  # Assuming unit area for simplicity
b = np.sqrt(AR * S)  # Span calculated from aspect ratio and area
c = S / b  # Chord length
rho = 1.225  # Density of air (kg/m^3)
V = 1  # Freestream velocity (m/s)
#define discretization of the spanwise direction
nSpanwise = 25
#define even spanwise discretization
y = np.linspace(0,1,nSpanwise+1)
#define cosine spacing to have better resolution close to the root
y_cos = 0.5-0.5*np.cos(y*np.pi)
#define control points for each cos spacing in exactly the middle of the spanwise discretization
y_control = (y_cos[1:] - y_cos[:-1])/2 + y_cos[:-1]
y_control_length = y_cos[1:] - y_cos[:-1]
#define bound vortex of strength Gamma_i, constant over the segment but varying from segment to segment
Gamma = np.zeros((nSpanwise))

# Calculate influence coefficients matrix
A = np.zeros((nSpanwise, nSpanwise))
for i in range(nSpanwise):
    for j in range(nSpanwise):
        A[i, j] = -1 / (4 * np.pi * (y_control[i] - y_cos[j])) + 1 / (4 * np.pi * (y_control[i] - y_cos[j + 1]))

# Iterative solution for circulation strengths
tolerance = 1e-6  # Convergence tolerance
max_iterations = 1000
for iteration in range(max_iterations):
    w = np.dot(A, Gamma)
    v_new = np.sqrt(V**2 + w**2 - 2 * V * w * np.cos(alpha))
    alpha_i = np.arccos((w**2 - V**2 - v_new**2)/(-2 * V * v_new))  # Calculate induced angles of attack (alpha_i)
    alpha_new = alpha - alpha_i
    Gamma_new = 0.5 * V * Cl(alpha_new) * c # Update RHS with induced angles
    # Check for convergence
    if np.linalg.norm(Gamma_new - Gamma) < tolerance:
        print("Converged after", iteration, "iterations")
        Gamma = Gamma_new
        break
    Gamma = Gamma_new

Cl_new = 2 * Gamma / (v_new * c)
plt.plot(y_control, Cl_new)
plt.show()
