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
'''Gamma = np.ones(int((nSpanwise+1)/2))

# Calculate influence coefficients matrix
A = np.zeros((int((nSpanwise+1)/2), int((nSpanwise+1)/2)))
for i in range(int((nSpanwise+1)/2)):
    for j in range(int((nSpanwise+1)/2)):
        A[i, j] = -1 / (4 * np.pi * (y_control[i] - y_cos[j])) + 1 / (4 * np.pi * (y_control[i] - y_cos[-(j + 1)]))

# Iterative solution for circulation strengths

tolerance = 1e-6  # Convergence tolerance
max_iterations = 100
for iteration in range(max_iterations):
    Cl_new = 2 * Gamma / V
    alpha_new = np.arcsin(Cl_new / (2 * np.pi))
    alpha_i = alpha - alpha_new
    w = np.tan(alpha_i) * V * np.cos(alpha)
    new_Gamma = np.linalg.solve(A, w)
    alpha_i = np.arctan(np.dot(A, Gamma) / V)  # Calculate induced angles of attack (alpha_i)
    Cl_new = Cl(alpha - alpha_i)  # Calculate new lift coefficients
    RHS = 0.5 * V  * Cl_new  # Update RHS with induced angles
    new_Gamma = np.zeros(int((nSpanwise+1)/2))
    for i in range(int((nSpanwise+1)/2)):
        new_Gamma[i] = RHS[i] - RHS[i - 1]
    # Check for convergence
    if np.linalg.norm(new_Gamma - Gamma) < tolerance:
        print("Converged after", iteration, "iterations")
        Gamma = new_Gamma
        break
    Gamma = new_Gamma

Cl_new = np.dot(A, Gamma)
print(Cl_new)
print("Cl_new:", Cl_new[:-1][::-1])
Cl_full = np.hstack((Cl_new, Cl_new[:-1][::-1]))
plt.plot(y_control, Cl_full)
plt.show()'''
'''# Calculate total lift using the trapezoidal rule for numerical integration
L = rho * V * np.trapz(Gamma, x=b * y_cos[:-1])

# Calculate lift coefficient
Cl_total = L / (0.5 * rho * V**2 * S)

print("Circulation strengths (Gamma):", Gamma)
print("Total lift coefficient (Cl_total):", Cl_total)'''



'''#Make definition to calculate the lift based on the spanwise discretization with the help of the lifting line theory
def LiftingLineTheory(Cl, alpha, y, c, Uinf, r, NBlades, Radius):
    #initialize the variables
    sigma = NBlades*c/(2*np.pi*r)
    a = 1/3 # induction factor
    a_old = 0
    epsilon = 1e-5
    Niterations = 1000
    for i in range(Niterations):
        #calculate the angle of attack
        alpha_i = alpha + a * 57.3
        #calculate the lift coefficient
        Cl_i = Cl(alpha_i)
        #calculate the lift
        F = Cl_i * 0.5 * Uinf**2 * c
        #calculate the induced velocity
        Vinduced = Uinf * (1-a)
        #calculate the local inflow angle
        phi = np.arctan(Vinduced/Uinf)
        #calculate the change in angle of attack
        dalpha = -3/2*Cl_i/(np.pi*AR)*a
        #calculate the change in induction factor
        da = 1/(4*np.pi)*((F/(2*np.pi*r)) / (0.5 * Uinf**2) - 1)
        #calculate the new induction factor
        a = a + da
        #calculate the error
        error = abs(a - a_old)
        #update the old induction factor
        a_old = a
        if error < epsilon:
            break
    return F, a, phi, alpha_i, dalpha
'''
