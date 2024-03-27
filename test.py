# import necessary libraries
import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd


data = pd.read_excel('polars.xlsx')
polar_cl = data['Cl'].to_numpy()
polar_cd = data['Cd'].to_numpy()
polar_alpha = data['Alfa'].to_numpy()
  
    
def ainduction(CT):
    a = np.zeros(np.shape(CT))
    CT1=1.816
    CT2=2*np.sqrt(CT1)-CT1
    a[CT>=CT2] = 1 + (CT[CT>=CT2]-CT1)/(4*(np.sqrt(CT1)-1))
    a[CT<CT2] = 0.5-0.5*np.sqrt(1-CT[CT<CT2])
    return a


def PrandtlTipRootCorrection(r_R, rootradius_R, tipradius_R, TSR, NBlades, axial_induction):
    temp1 = -NBlades/2*(tipradius_R-r_R)/r_R*np.sqrt( 1+ ((TSR*r_R)**2)/((1-axial_induction)**2))
    Ftip = np.array(2/np.pi*np.arccos(np.exp(temp1)))
    Ftip[np.isnan(Ftip)] = 0
    temp1 = NBlades/2*(rootradius_R-r_R)/r_R*np.sqrt( 1+ ((TSR*r_R)**2)/((1-axial_induction)**2))
    Froot = np.array(2/np.pi*np.arccos(np.exp(temp1)))
    Froot[np.isnan(Froot)] = 0
    return Froot*Ftip, Ftip, Froot


# define function to determine load in the blade element
def loadBladeElement(vnorm, vtan, r_R, chord, twist, polar_alpha, polar_cl, polar_cd):
    """
    calculates the load in the blade element
    """
    vmag2 = vnorm**2 + vtan**2
    inflowangle = np.arctan2(vnorm,vtan)
    alpha = twist + inflowangle*180/np.pi
    cl = np.interp(alpha, polar_alpha, polar_cl)
    cd = np.interp(alpha, polar_alpha, polar_cd)
    lift = 0.5*vmag2*cl*chord
    drag = 0.5*vmag2*cd*chord
    fnorm = lift*np.cos(inflowangle)+drag*np.sin(inflowangle)
    ftan = lift*np.sin(inflowangle)-drag*np.cos(inflowangle)
    gamma = 0.5*np.sqrt(vmag2)*cl*chord
    return fnorm , ftan, gamma, inflowangle*180/np.pi, alpha

def solveStreamtube(Uinf, r1_R, r2_R, rootradius_R, tipradius_R , Omega, Radius, NBlades, chord, twist, polar_alpha, polar_cl, polar_cd ):

    Area = np.pi*((r2_R*Radius)**2-(r1_R*Radius)**2) #  area streamtube
    r_R = (r1_R+r2_R)/2 # centroide
    # initiatlize variables
    a = 0.0 # axial induction
    aline = 0.0 # tangential induction factor
    
    Niterations = 100
    Erroriterations =0.000001 # error limit for iteration rpocess, in absolute value of induction
    
    for i in range(Niterations):
        # ///////////////////////////////////////////////////////////////////////
        # // this is the block "Calculate velocity and loads at blade element"
        # ///////////////////////////////////////////////////////////////////////
        Urotor = Uinf*(1-a) # axial velocity at rotor
        Utan = (1+aline)*Omega*r_R*Radius # tangential velocity at rotor
        # calculate loads in blade segment in 2D (N/m)
        fnorm, ftan, gamma, phi, AoA = loadBladeElement(Urotor, Utan, r_R,chord, twist, polar_alpha, polar_cl, polar_cd)
        load3Daxial =fnorm*Radius*(r2_R-r1_R)*NBlades # 3D force in axial direction
        # load3Dtan =loads[1]*Radius*(r2_R-r1_R)*NBlades # 3D force in azimuthal/tangential direction (not used here)
      
        # ///////////////////////////////////////////////////////////////////////
        # //the block "Calculate velocity and loads at blade element" is done
        # ///////////////////////////////////////////////////////////////////////

        # ///////////////////////////////////////////////////////////////////////
        # // this is the block "Calculate new estimate of axial and azimuthal induction"
        # ///////////////////////////////////////////////////////////////////////
        # // calculate thrust coefficient at the streamtube 
        CT = load3Daxial/(0.5*Area*Uinf**2)
        
        # calculate new axial induction, accounting for Glauert's correction
        anew =  ainduction(CT)
        
        # correct new axial induction with Prandtl's correction
        Prandtl, Prandtltip, Prandtlroot = PrandtlTipRootCorrection(r_R, rootradius_R, tipradius_R, Omega*Radius/Uinf, NBlades, anew);
        if (Prandtl < 0.00001): 
            Prandtl = 0.00001 # avoid divide by zero
        anew = anew/Prandtl # correct estimate of axial induction
        a = 0.75*a+0.25*anew # for improving convergence, weigh current and previous iteration of axial induction
        if (a>0.95):
            a=0.95

        aline = ftan*NBlades/(2*np.pi*Uinf*(1-a)*Omega*2*(r_R*Radius)**2)
        aline =aline/Prandtl 
        
        if (np.abs(a-anew) < Erroriterations): 
            # print("iterations")
            # print(i)
            break

    return [a , aline, r_R, fnorm , ftan, gamma, phi, AoA]

# define the blade geometry
delta_r_R = .01
r_R = np.arange(0.2, 1+delta_r_R/2, delta_r_R)
print(r_R)


# blade shape
pitch = 2 # degrees
chord_distribution = 3*(1-r_R)+1 # meters
twist_distribution = -14*(1-r_R)+pitch # degrees
# define flow conditions
Uinf = 10 # unperturbed wind speed in m/s
NBlades = 3

TipLocation_R =  1
RootLocation_R =  0.2

Radius = 50
TSR = [6, 8, 10] # tip speed ratio
final_results = np.zeros([len(r_R)-1,8,3])
for j in range(len(TSR)):
    Omega = Uinf*TSR[j]/Radius

    results =np.zeros([len(r_R)-1,8]) 

    for i in range(len(r_R)-1):
        chord = np.interp((r_R[i]+r_R[i+1])/2, r_R, chord_distribution)
        twist = np.interp((r_R[i]+r_R[i+1])/2, r_R, twist_distribution)
        
        results[i,:] = solveStreamtube(Uinf, r_R[i], r_R[i+1], RootLocation_R, TipLocation_R , Omega, Radius, NBlades, chord, twist, polar_alpha, polar_cl, polar_cd )
    final_results[:,:,j] = results[:,:]

def plot_alpha_rR(final_results, TSR, save=False):
    for i in range(len(TSR)):
        plt.plot(final_results[:,2,i], final_results[:,7,i], label = 'TSR = '+str(TSR[i]))
    plt.ylabel(r'$\alpha$')
    plt.xlabel('r/R')
    plt.grid()
    plt.legend()
    if save:
        plt.savefig('alpha_rR.png')
        plt.cla()
    else:
        plt.show()

def plot_phi_rR(final_results, TSR, save=False):
    for i in range(len(TSR)):
        plt.plot(final_results[:,2,i], final_results[:,6,i], label = 'TSR = '+str(TSR[i]))
    plt.ylabel(r'$\phi$')
    plt.xlabel('r/R')
    plt.grid()
    plt.legend()
    if save:
        plt.savefig('phi_rR.png')
        plt.cla()
    else:
        plt.show()

def plot_a_rR(final_results, TSR, save=False):
    for i in range(len(TSR)):
        plt.plot(final_results[:,2,i], final_results[:,0,i], label = 'TSR = '+str(TSR[i]))
    plt.ylabel(r'$a$')
    plt.xlabel('r/R')
    plt.grid()
    plt.legend()
    if save:
        plt.savefig('a_rR.png')
        plt.cla()
    else:
        plt.show()
def plot_aprime_rR(final_results, TSR, save=False):
    for i in range(len(TSR)):
        plt.plot(final_results[:,2,i], final_results[:,1,i], label = 'TSR = '+str(TSR[i]))
    plt.ylabel(r'$a\prime$')
    plt.xlabel('r/R')
    plt.grid()
    plt.legend()
    if save:
        plt.savefig('aprime_rR.png')
        plt.cla()
    else:
        plt.show()

def plot_fnorm_rR(final_results, TSR, save=False):
    for i in range(len(TSR)):
        plt.plot(final_results[:,2,i], final_results[:,3,i]/(0.5*Uinf**2*Radius), label = 'TSR = '+str(TSR[i]))
    plt.ylabel(r'$F_{norm}$')
    plt.xlabel('r/R')
    plt.grid()
    plt.legend()
    if save:
        plt.savefig('fnorm_rR.png')
        plt.cla()
    else:
        plt.show()

def plot_ftan_rR(final_results, TSR, save=False):
    for i in range(len(TSR)):
        plt.plot(final_results[:,2,i], final_results[:,4,i]/(0.5*Uinf**2*Radius), label = 'TSR = '+str(TSR[i]))
    plt.ylabel(r'$F_{tan}$')
    plt.xlabel('r/R')
    plt.grid()
    plt.legend()
    if save:
        plt.savefig('ftan_rR.png')
        plt.cla()
    else:
        plt.show()

save = True

plot_alpha_rR(final_results, TSR, save)
plot_phi_rR(final_results, TSR, save)
plot_a_rR(final_results, TSR, save)
plot_aprime_rR(final_results, TSR, save)
plot_fnorm_rR(final_results, TSR, save)
plot_ftan_rR(final_results, TSR, save)

