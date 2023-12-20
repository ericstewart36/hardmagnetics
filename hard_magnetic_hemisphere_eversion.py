"""
Code for modeling hard-magnetic magneto-viscoelastic snap-through.

- with the model comprising:
    > Large deformation compressible Neo-Hookean elasticity
    > Finite deformation viscoelasticity through a generalized Maxwell
      model with 3 branches, with evolution of internal variables in the
      style of Green and Tobolsky (1946) and Linder et al. (2011).
    > Inertial effects using the Newmark kinematic relations.
    > Magnetic stresses due to magneto-quasistatic interactions between
        - Permanently magnetized particles (m_rem = b_rem/mu0) embedded in a
          mechanically soft matrix, and 
        - an externally applied magnetic flux density (b_app).       
    
- Under suitable assumptions described in the paper, magneto-quasistatics is 
  satisfied automatically and the only necessary numerical degree of freedom is 
  the displacement vector u.
    
- To aid in modeling the near-incompressibility we also introduce a 
  pressure-like DOF p which satisfies (J-1) - p/K = 0.
    > This is the classical (u,p) approach.
    
- Basic units:
    > Length: mm
    >   Time: s
    >   Mass: kg
    > Charge: kC
    
- Derived units: 
    >             Force: mN
    >          Pressure: kPa
    >           Current: kA
    > Mag. flux density: mT
    
    Eric M. Stewart      
   (ericstew@mit.edu)   
    
      Spring 2023 
   
"""

# FEniCS package
from dolfin import *
# NumPy for arrays and array operations
import numpy as np
# MatPlotLib for plotting
import matplotlib.pyplot as plt
# Current time package
from datetime import datetime


# Set level of detail for log messages (integer)
# 
# Guide:
# CRITICAL  = 50, // errors that may lead to data corruption
# ERROR     = 40, // things that HAVE gone wrong
# WARNING   = 30, // things that MAY go wrong later
# INFO      = 20, // information of general interest (includes solver info)
# PROGRESS  = 16, // what's happening (broadly)
# TRACE     = 13, // what's happening (in detail)
# DBG       = 10  // sundry
#
set_log_level(30)

# Global FEniCS parameters:
# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"
parameters["form_compiler"]["quadrature_degree"] = 4

'''''''''''''''''''''
DEFINE GEOMETRY
'''''''''''''''''''''

scaleX = 30.0 # Radius of outer flange
scaleY = 18.0 # Radius of hemispherical shell itself

# Initialize an empty mesh object
mesh = Mesh()


# Read the *.xdmf file data into mesh object
with XDMFFile("hemisphere_9k.xdmf") as infile:
    infile.read(mesh)


# This says "spatial coordinates" but is really the referential coordinates,
# since the mesh does not convect in FEniCS.
x = SpatialCoordinate(mesh) 


'''''''''''''''''''''
     SUBDOMAINS
'''''''''''''''''''''


#----------------------------------------------------------
# Define the mesh subdomains, used for applying BCs and 
#  spatially-varying material properties.

tol = 0.5*mesh.hmin() # half as wide as the smallest element in the mesh

#Pick up on the boundary entities of the created mesh
class OuterEdge(SubDomain):
    def inside(self, x, on_boundary):
        return near(np.sqrt(x[0]*x[0] + x[2]*x[2]),scaleX, tol) and on_boundary


# Mark boundary subdomians
facets = MeshFunction("size_t", mesh, 2)
facets.set_all(0)
DomainBoundary().mark(facets, 1)  # First, mark all boundaries with common index

# Next mark specific boundaries
OuterEdge().mark(facets,3)

# Define a ds measure for each face, necessary for applying traction BCs.
ds = Measure('ds', domain=mesh, subdomain_data=facets)

'''''''''''''''''''''
MATERIAL PARAMETERS
'''''''''''''''''''''

# Gent elasticity
Gshear0 = Constant(150.0) # kPa
Kbulk   = Constant(1e3*Gshear0) # Nearly-incompressible

# Mass density
rho = Constant(2.000e-6) # 1.75e3 kg/m^3 = 1.75e-6 kg/mm^3

# Vacuum permeability
mu0 = Constant(1.256e-6*1e9) # mN / mA^2

# Remanent magnetic flux density vector
#b_rem_mag = 100.0 # mT
b_rem_mag = 80.0*mu0/1e3 # 80 kA/m converted to mT
b_rem   = Expression(("0", "( pow(pow(x[0],2) + pow(x[2],2), 0.5) < r+0.75 ) ? b : ( (t<0.00020)? tol : 0.0) ", "0"), \
                     r=scaleY, b=float(b_rem_mag), t=0.0, tol=float(b_rem_mag)/200, degree=1)
                        # I found that we need some non-zero magnetization everywhere 
                        # for initial convergence. Here, the unmagnetized regions 
                        # have 1/200 of the strength of the magnetized regions for 0.2 ms,
                        # then 0 magnetization for all times after.


# Max applied magnetic flux density magnitude
b_app_max = Constant(200) 
angle_imperf = -1.0*np.pi/180 # imperfection in degrees, converted to radians

# alpha-method parameters
alpha   = Constant(0.0) # 
gamma   = Constant(0.5+alpha)
beta    = Constant((gamma+0.5)**2/4.)

# Visco dissipation switch, 0=dissipative,  1=~lossless
disableDissipation = Constant(0.0)
#
# When enabled, this switch sets the relaxation times arbitrarily high, 
# so that the stiffness remains the same but no energy is dissipated 
# because the tensor variables A_i are constant.


# Viscoelasticity parameters
#
Gneq_1  = Constant(100.00)    #  Non-equilibrium shear modulus, kPa
tau_1   = Constant(0.01)      #  relaxation time, s

# Set relaxation times arbitrarily high if visco dissipation is off
tau_1 = tau_1 + disableDissipation/DOLFIN_EPS

#Simulation time related params
t = 0.0         # initialization of time  
step2_dt     = 0.003/100

# Compiler variable for time step
dk = Constant(step2_dt)


'''''''''''''''''''''
   FUNCTION SPACES
'''''''''''''''''''''

# Define function space, scalar
U2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
T1 = TensorElement("Lagrange", mesh.ufl_cell(), 1, symmetry=True, shape=(3,3)) # tensor internal variable 

# DOFs
TH = MixedElement([U2, P1])
ME = FunctionSpace(mesh, TH) # Total space for all DOFs

W = FunctionSpace(mesh,P1)   # Scalar space for visualization later
W2 = FunctionSpace(mesh,U2)   # Vector space for visualization later
W3 = FunctionSpace(mesh,T1)   # Tensor space for visualization later

# Define test functions in weak form
dw = TrialFunction(ME)                                   
(u_test, p_test)  = TestFunctions(ME)    # Test function

# Define actual functions with the required DOFs
w = Function(ME)
(u, p) = split(w)    # current DOFs

# A copy of functions to store values in last step for time-stepping.
w_old = Function(ME)
(u_old, p_old) = split(w_old)   # old DOFs

# initialize old velocity and acceleration fields
v_old = Function(W2)
a_old = Function(W2)

# ############ Initialization of the tensorial internal variables
# 
# Internal tensor variables for A1
#
A1 = Function(W3)
#
# We need to initialize the field A1_old to identity.
#
# Tensor functions for internal variables at previous step
A1_old = Function(W3)
#
# Assign identity as initial value for the above function
A1_old.assign(project(Identity(3), W3))

'''''''''''''''''''''
     SUBROUTINES
'''''''''''''''''''''

def F_calc(u):
    dim = len(u)
    Id = Identity(dim) # Identity tensor
    F = Id + grad(u) # 3D Deformation gradient
    return F # Full 3D F

def safe_sqrt(x):
    return sqrt(x + DOLFIN_EPS)

def right_decomp(F):
        
    Id = Identity(3)
    
    # Compute U and R using the methods of Hoger and Carlson (1984) 
    
    # invariants of C
    C = F.T*F
    #
    I1C = tr(C)
    I2C = (1/2)*(tr(C)**2 - tr(C*C))
    I3C = det(C)

    # intermediate quantity \lambda
    #
    # Have to ensure the argument to acos( ) is in [-1, 1]
    arg = (2*(I1C**3) -9*I1C*I2C + 27*I3C)/( 2*safe_sqrt((I1C*I1C - 3*I2C)**(3)) ) 
    arg2 = conditional(gt(arg, 1), 1, arg)
    arg3 = conditional(lt(arg2, -1), -1, arg)
    lambdaU = safe_sqrt(I1C + 2*safe_sqrt(I1C**2 - 3*I2C)*cos((1/3)*acos(arg3) ))/sqrt(3)
    
    # U invariants
    I3U = safe_sqrt(I3C)
    I2U = safe_sqrt(I3C)/lambdaU + safe_sqrt(I1C*(lambdaU**2) - lambdaU**4 + 2*safe_sqrt(I3C)*lambdaU)
    I1U = lambdaU + safe_sqrt(I1C - lambdaU**2 + 2*safe_sqrt(I3C)/lambdaU )
    
    # intermediate quantity \Delta U
    deltaU = I1U*I2U - I3U
    
    # final expression for U^{-1} tensor
    Uinv = ((I3U*deltaU)**(-1))*( \
                    + (I1U)*(C*C) \
                    - ( I3U + I1U**3 - 2*I1U*I2U)*C \
                    + (I1U*(I2U**2)  - I3U*(I1U**2) - I3U*I2U)*Id )
        
    R = F*Uinv
    U = R.T*F

    return R, U


def Piola(F, R, U, p, b_app, A1):
    
    Id = Identity(3)
    
    J = det(F)
    C = F.T*F
    Cdis = J**(-2/3)*C
     
    # Calculate the derivative dRdF after Chen and Wheeler (1992)
    #
    Y = tr(U)*Id - U # helper tensor Y 
    #
    Lmat = -outer(b_app, b_rem)/mu0 # dRdF will act on this tensor
    #
    T_mag = R*Y*(R.T*Lmat - Lmat.T*R)*Y/det(Y)
    
    # The viscous Piola stress
    #
    T_visc = J**(-2/3)*(Gneq_1*F*(A1 - (1/3)*inner(Cdis, A1)*inv(Cdis)) )
    
    # Piola stress
    TR = J**(-2/3)*Gshear0*(F - 1/3*tr(C)*inv(F.T)) \
        + J*p*inv(F.T) + T_mag + T_visc
    
    return TR

    
# Update formula for acceleration
# a = 1/(2*beta)*((u - u0 - v0*dt)/(0.5*dt*dt) - (1-2*beta)*a0)
def update_a(u, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dk
        beta_ = beta
    else:
        dt_ = float(dk)
        beta_ = float(beta)
    return (u-u_old-dt_*v_old)/beta_/dt_**2 - (1-2*beta_)/2/beta_*a_old

# Update formula for velocity
# v = dt * ((1-gamma)*a0 + gamma*a) + v0
def update_v(a, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dk
        gamma_ = gamma
    else:
        dt_ = float(dk)
        gamma_ = float(gamma)
    return v_old + dt_*((1-gamma_)*a_old + gamma_*a)

def update_fields(u_proj, u_proj_old, v_old, a_old):
    """Update fields at the end of each time step."""

    # Get vectors (references)
    u_vec, u0_vec  = u_proj.vector(), u_proj_old.vector()
    v0_vec, a0_vec = v_old.vector(), a_old.vector()

    # use update functions using vector arguments
    a_vec = update_a(u_vec, u0_vec, v0_vec, a0_vec, ufl=False)
    v_vec = update_v(a_vec, u0_vec, v0_vec, a0_vec, ufl=False)

    # Update (u_old <- u)
    v_old.vector()[:], a_old.vector()[:] = v_vec, a_vec
    #u_old.vector()[:] = u_proj.vector()

def avg(x_old, x_new, alpha):
    return alpha*x_old + (1-alpha)*x_new

# Macaulay bracket function
def ppos(x):
    return (x+abs(x))/2.


# Slightly faster projection function for linear fields
# Based on Jermey Bleyer's function of the same name, cf. e.g.
# https://gitlab.enpc.fr/jeremy.bleyer/comet-fenics/-/blob/1028fd8438e2b23a69ae78d7d46ec810a5b8e3da/examples/nonlinear_materials/vonMises_plasticity.py.rst
#
def local_project(v, V, u=None):
    if V.ufl_element().degree() ==1:
        dv = TrialFunction(V)
        v_ = TestFunction(V)
        a_proj = inner(dv, v_)*dx
        b_proj = inner(v, v_)*dx
        Lsolver = LocalSolver(a_proj, b_proj)
        Lsolver.factorize
        if u is None:
            u = Function(V)
            Lsolver.solve_local_rhs(u)
            return u
        else:
            Lsolver.solve_local_rhs(u)
            return
    else:
        u = project(v,V)
        return u
    
'''''''''''''''''''''''''''''''''''''''''
  KINEMATICS & CONSTITUTIVE RELATIONS
'''''''''''''''''''''''''''''''''''''''''

# Newmark-beta kinematical update
a_new = update_a(u, u_old, v_old, a_old, ufl=True)
v_new = update_v(a_new, u_old, v_old, a_old, ufl=True)

# get avg fields for generalized-alpha method
u_avg  = avg(u_old, u, alpha)
p_avg  = avg(p_old, p, alpha)
v_avg = avg(v_old, v_new, alpha)

# Kinematics
F = F_calc(u_avg)
C = F.T*F
Ci = inv(C)
F_old = F_calc(u_old)
J = det(F)
J_old = det(F_old)
Cdis_3D = J**(-2/3)*C

# Right polar decomposition.
R_old, U_old = right_decomp(F_old)
R, U = right_decomp(F)

# Discretized evolution equations for updating A1
A1 = (1/(1+dk/tau_1))*(A1_old + (dk/tau_1)*inv(Cdis_3D))

'''''''''''''''''''''''
BOUNDARY CONDITIONS
'''''''''''''''''''''''  

# Mechanical BCs
bcs_1  = DirichletBC(ME.sub(0).sub(0),0.0,facets,3) # right face x-fix
bcs_2  = DirichletBC(ME.sub(0).sub(1),0.0,facets,3) # right face y-fix
bcs_3  = DirichletBC(ME.sub(0).sub(2),0.0,facets,3) # right face z-fix

bcs = [bcs_1, bcs_2, bcs_3]

# Time-varying applied magnetic flux density (within whole domain, not a surface BC)
step2_time   = 0.2    # Total application time
#
b_app_mag = Expression("magnitude*t",\
                   magnitude=b_app_max, t=0.0, degree=1)

# Apply imperfection angle
x_mag = np.sin(angle_imperf)
y_mag = np.cos(angle_imperf)
z_mag = 0.0

b_app = as_vector([-x_mag*b_app_mag, -y_mag*b_app_mag, 0.0])


'''''''''''''''''''''''
       WEAK FORMS
'''''''''''''''''''''''

# Equation of motion
L0 = inner(Piola(F, R, U, p_avg, b_app, A1), grad(u_test))*dx + inner(rho*a_new, u_test)*dx 
   
# Pressure penalty term
L1 = inner(((J-1) - p_avg/Kbulk), p_test)*dx
   
# Total weak form
L = (1/Gshear0)*L0 + L1

# Automatic differentiation tangent: 
#
#   When computing the Piola stress for the tangent, we approximate R and U as
#   F and Identity(3) respectively, in order to avoid messy terms in the 
#   derivative of the U^{-1} calculation which cause numerical convergence issues.
#
#   Importantly, the residuals are still enforced with the current R and U.
#   Changing the Jacobian in this manner only changes the rate of convergence 
#   of the Newton-Raphson solver, and not the ultimate result.
#
L0_prime = inner(Piola(F, F, Identity(3), p_avg, b_app, A1), grad(u_test))*dx + inner(rho*a_new, u_test)*dx 
#
L_prime  = (1/Gshear0)*L0_prime + L1
#
a = derivative(L_prime, w, dw)


'''''''''''''''''''''
    RUN ANALYSIS
'''''''''''''''''''''

# Output file setup
file_results = XDMFFile("results/hard_magnetic_hemisphere_eversion.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

# Give fields descriptive names
u_v = w_old.sub(0)
u_v.rename("displacement","")

p_v = w_old.sub(1)
p_v.rename("p", "")

b_out       = np.zeros(100000)
disp_out    = np.zeros(100000)
time_out    = np.zeros(100000)

ii=0

CoupledProblem = NonlinearVariationalProblem(L, w, bcs, J=a)

# Set up the non-linear solver
solver  = NonlinearVariationalSolver(CoupledProblem)

# Solver parameters
prm = solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = 'mumps'  
prm['newton_solver']['absolute_tolerance'] = 1.E-6
prm['newton_solver']['relative_tolerance'] = 1.E-6
prm['newton_solver']['maximum_iterations'] = 30

# function to write results to XDMF at time t
def writeResults(t):

    # Displacement, pressure penalty term
    file_results.write(u_v,t)
    file_results.write(p_v,t)
    
    # Write the effective stretch
    lambda_eff = safe_sqrt(tr(Cdis_3D)/3)
    lambda_eff_v = project(lambda_eff, FunctionSpace(mesh, "CG", 1))
    lambda_eff_v.rename("lambda_eff","")
    file_results.write(lambda_eff_v,t)
    
    # Write J
    J_v = project(J, FunctionSpace(mesh, "CG", 1))
    J_v.rename("J","")
    file_results.write(J_v,t)
    
    # Write the spatial m_rem
    m_Vis  = R*b_rem/det(F)/mu0*1000 # units of kA/m
    m_v = local_project(m_Vis, VectorFunctionSpace(mesh, "CG", 1))
    m_v.rename("m_rem","")
    #
    file_results.write(m_v,t)
    
    # Write the magnitude of R*m^rem_mat
    Rm_rem  = R*b_rem/mu0*1000 # units of kA/m
    m_mag  = safe_sqrt(dot(Rm_rem, Rm_rem))
    m_mag_v = project(m_mag, FunctionSpace(mesh, "CG", 1))
    m_mag_v.rename("m_mag","")
    #
    file_results.write(m_mag_v,t)

# Write initial state at t=0
writeResults(t=0.0)

# Give the step a descriptive name
step = "Eversion"

while (round(t,4) <= round(2*step2_time + 0.1,4)):
    
    # update the time and time-dependent BCs
    t += step2_dt
    b_app_mag.t = 1.0 # constant b-field 
    
    if t>0.1:
        b_app_mag.t = 0.0 # turn off b-field from 0.1 s to 0.2 s
        if t>0.2:
            b_app_mag.t = -0.4 # reverse b-field after 0.2 s
            if t>0.3:
                b_app_mag.t = 0.0 # turn off b-field again after 0.3 s

    # Solve the problem
    (iter, converged) = solver.solve()
     
    # increment counter
    ii = ii + 1
    
    # Output time histories
    b_out[ii] = float(b_app_max)
    if t>0.1:
        b_out[ii] = 0.0 # turn off b-field from 0.1 s to 0.2 s
        if t>0.2:
            b_out[ii] =  -0.4*float(b_app_max) # reverse b-field after 0.2 s
            if t>0.3:
                b_out[ii] = 0.0 # turn off b-field again after 0.3 s
    disp_out[ii] = w(0, scaleY, 0)[1]
    time_out[ii] = t
    
    # Update fields for next step
    u_proj = project(u, W2)
    u_proj_old = project(u_old, W2)
    update_fields(u_proj, u_proj_old, v_old, a_old)
    w_old.vector()[:] = w.vector()
    
    # Update state variables
    A1_old.assign(local_project(A1, W3))
    
    # Print the calculation progress periodically.
    if ii%5==0:
        print("Step: {}   |   Simulation Time: {} s  |     Iterations: {}".format(step, t, iter))
        print()
        
    # Writing to *.xdmf takes considerable time, so here I have limited the calls
    # to writeResults(t) to once every 50 steps. For the movies included in the paper
    # I called writeResults(t) once every 5 steps.
    #
    if ii%50==0:
        # Write results to *.xdmf at current time
        writeResults(t)

'''''''''''''''''''''
    VISUALIZATION
'''''''''''''''''''''
        
# Set up font size, initialize colors array
font = {'size'   : 16}
plt.rc('font', **font)
#
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# only plot as far as time_out has time history for.
ind  = np.argmax(time_out)

plt.figure()
plt.axvline(0, c='k', linewidth=1.)
plt.axhline(0, c='k', linewidth=1.)
plt.plot(time_out[0:ind], disp_out[0:ind], linewidth=2.5, \
         color=colors[3])
plt.axis('tight')
plt.xlabel(r"Time (s)")
plt.ylabel(r"$u_2$ at point A (mm)")
plt.grid(linestyle="--", linewidth=0.5, color='b')
#plt.ylim(-4, 12)
#plt.ylim(5,10)
plt.xlim(0.0,2*step2_time + 0.1)
#plt.xlim(0.9,1.4)

# save figure to file
fig = plt.gcf()
fig.set_size_inches(6, 4)
plt.tight_layout()
plt.savefig("hard_magnetic_eversion.png", dpi=600)
