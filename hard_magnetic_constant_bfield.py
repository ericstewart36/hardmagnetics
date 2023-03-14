"""
Code for modeling hard-magnetic magneto-viscoelastic snap-through.

- with the model comprising:
    > Large deformation compressible Neo-Hookean elasticity
    > Finite deformation viscoelasticity through a generalized Maxwell
      model with 3 branches, with evolution of internal variables in the
      style of Green and Tobolsky (1946) and Linder et al. (2011).
    > Inertial effects using the Newmark kinematic relations.
    > Body couples due to magneto-quasistatic interactions between
        - Permanently magnetized particles (b_rem) embedded in a mechanically soft matrix, and 
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
parameters["form_compiler"]["quadrature_degree"] = 2

'''''''''''''''''''''
DEFINE GEOMETRY
'''''''''''''''''''''

# Overall dimensions of rectangular prism device
scaleX = 120.47 #60.26 # mm
scaleY = 2.5   # mm
scaleZ = 20.0  # mm

# N number of elements in each direction    
Xelem = 31
Yelem = 2
Zelem = 3
  
# Define a uniformly spaced box mesh
mesh = BoxMesh(Point(0.0, 0.0, 0.0),Point(scaleX,scaleY, scaleZ),Xelem, Yelem, Zelem)

# Add an initial imperfection to control buckling mode
imperf = 0.01 # mm 

# Map the coordinates of the uniform box mesh to the biased spacing
xOrig = mesh.coordinates()
xMap1 = np.zeros((len(xOrig),3))

# Mapping functions
for i in range(0,len(xMap1)):

    xMap1[i,0] = xOrig[i,0] 
    xMap1[i,1] = xOrig[i,1] + (imperf/2.0)*(1.0 - np.cos(2*np.pi*xOrig[i,0]/scaleX)) 
    xMap1[i,2] = xOrig[i,2] 

mesh.coordinates()[:] = xMap1

# This says "spatial coordinates" but is really the referential coordinates,
# since the mesh does not convect in FEniCS.
x = SpatialCoordinate(mesh) 


'''''''''''''''''''''
     SUBDOMAINS
'''''''''''''''''''''

#----------------------------------------------------------
# Define the mesh subdomains, used for applying BCs and 
#  spatially-varying material properties.

tol = 1e-12

#Pick up on the boundary entities of the created mesh
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],0.0, tol) and on_boundary
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],scaleX, tol) and on_boundary
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.0, tol) and on_boundary
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],scaleY, tol) and on_boundary  
class Back(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2],0.0, tol) and on_boundary
class Front(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2],scaleZ, tol) and on_boundary


# Mark boundary subdomians
facets = MeshFunction("size_t", mesh, 2)
facets.set_all(0)
DomainBoundary().mark(facets, 1)  # First, mark all boundaries with common index

# Next mark sepcific boundaries
Left().mark(facets, 2)
Right().mark(facets,3)
Bottom().mark(facets, 4)
Top().mark(facets,5)
Back().mark(facets, 6)
Front().mark(facets,7)

# Define a ds measure for each face, necessary for applying traction BCs.
ds = Measure('ds', domain=mesh, subdomain_data=facets)

'''''''''''''''''''''
MATERIAL PARAMETERS
'''''''''''''''''''''

# Gent elasticity
Gshear0 = Constant(1400.0) # kPa
Kbulk   = Constant(1e3*Gshear0) # Nearly-incompressible

# Mass density
rho = Constant(2.000e-6) # 1.75e3 kg/m^3 = 1.75e-6 kg/mm^3

# Vacuum permeability
mu0 = Constant(1.256e-6*1e9) # mN / mA^2

# Remanent magnetic flux density vector
b_rem_mag = 67.5 # mT

# Spatially-varying remanent magnetization
b_rem = Expression(("(x[0]<scaleX/2) ? -b_rem_mag : b_rem_mag", "0.0",  "0.0"),\
                   b_rem_mag=b_rem_mag, scaleX = scaleX, degree=0) 

# Max applied magnetic flux density magnitude
b_app_max = Constant(7.0) #Constant(0.0094*Gshear0*mu0/b_rem_mag) 

# alpha-method parameters
alpha   = Constant(0.0) # Here alpha-method is not needed, set \alpha=0
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
Gneq_1  = Constant(500.00)    #  Non-equilibrium shear modulus, kPa
tau_1   = Constant(0.010)      #  relaxation time, s

# Set relaxation times arbitrarily high if visco dissipation is off
tau_1 = tau_1 + disableDissipation/DOLFIN_EPS

#Simulation time related params
t = 0.0         # initialization of time  

# total simulation time 
T_tot = 1.0e3 # s

# Float value of time step
dt = T_tot/50 # time step size, seconds

# Compiler variable for time step
dk = Constant(dt)

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

    
def Piola(F,p, b_app, A1):
    
    Id = Identity(3)
    J = det(F)
    
    C = F.T*F
    Cdis = J**(-2/3)*C
    I1 = tr(Cdis)
    #
    T_mag = -outer(b_app, b_rem)/mu0
    #
    T_visc = J**(-2/3)*(Gneq_1*F*(A1 - (1/3)*inner(Cdis, A1)*inv(Cdis)))
    
    # Piola stress
    TR = J**(-2/3)*Gshear0*(F - 1/3*tr(C)*inv(F.T))\
        + p*inv(F.T) + T_mag + T_visc
    
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

# Discretized evolution equations for updating A1
A1 = (1/(1+dk/tau_1))*(A1_old + (dk/tau_1)*inv(Cdis_3D))

'''''''''''''''''''''''
BOUNDARY CONDITIONS
'''''''''''''''''''''''  

# Homogeneous deformation mechanical BCs
bcs_a  = DirichletBC(ME.sub(0).sub(0),0.0,facets,2) # left face x-fix
bcs_a1 = DirichletBC(ME.sub(0).sub(1),0.0,facets,2) # left face y-fix
bcs_b  = DirichletBC(ME.sub(0).sub(2),0.0,facets,2) # left face z-fix

# Time-varying applied displacement (ramp then hold)
disp_tot = -0.47 #-0.26 # mm
disp_exp = Expression(("max(disp_tot*t/Tramp, disp_tot)"),
                  disp_tot=disp_tot, t=0, Tramp = T_tot, degree=1)

bcs_c  = DirichletBC(ME.sub(0).sub(0),disp_exp,facets,3) # right face x move
bcs_d  = DirichletBC(ME.sub(0).sub(1),0.0,facets,3) # right face y-fix
bcs_d1 = DirichletBC(ME.sub(0).sub(2),0.0,facets,3) # right face z-fix

bcs = [bcs_a, bcs_a1, bcs_b, bcs_c, bcs_d, bcs_d1]

# Time-varying applied magnetic flux density (within whole domain, not a surface BC)
step2_time   = 0.6     # Total ramp time
step2_dt     = 0.003
Nstep2       = step2_time/step2_dt
#

#b_app_mag = Expression(("magnitude*t"),
b_app_mag = Expression(("magnitude*t"),
                  magnitude = b_app_max, t=0, Tramp = step2_time, degree=1)
#
b_app = as_vector([0.0, b_app_mag, 0.0])



'''''''''''''''''''''''
       WEAK FORMS
'''''''''''''''''''''''

# Equation of motion
L0 = inner(Piola(F, p_avg, b_app, A1), grad(u_test))*dx + inner(rho*a_new, u_test)*dx 
   
# Pressure penalty term
L1 = inner(((J-1) - p_avg/Kbulk), p_test)*dx
   
# Total weak form
L = (1/Gshear0)*L0 + L1

# Automatic differentiation tangent:
a = derivative(L, w, dw)

'''''''''''''''''''''
    RUN ANALYSIS
'''''''''''''''''''''

# Output file setup
file_results = XDMFFile("results/hard_magnetic_constant_bfield.xdmf")
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
    
    # Write the spatial b_rem
    b_Vis  = F*b_rem
    b_v = local_project(b_Vis, VectorFunctionSpace(mesh, "Lagrange", 1))
    b_v.rename("b_rem","")
    #
    file_results.write(b_v,t)

# Write initial state at t=0
writeResults(t=0.0)

while (round(t,4) <= round(T_tot + step2_time,4)):

    # condition for second phase:
    if t+dt>T_tot:
        #
        step = "Snap"
        dt = step2_dt
        dk.assign(dt)
        t += dt
        #b_app_mag.t = t - T_tot
        b_app_mag.t = 1.0
    else: # updates for first phase:
        t += dt
        # update time-varying BCs
        disp_exp.t = t
        step = "Buckle"
        
    # increment time, counter
    ii = ii + 1
    
    # Solve the problem
    (iter, converged) = solver.solve()
    
    # Write results to *.xdmf at current time
    writeResults(t)
    
    # Output time histories
    if t+dt>T_tot:
        b_out[ii] = b_app_max*(t-T_tot)/step2_time
    else:
        b_out[ii] = 0.0
    disp_out[ii] = w(scaleX/2, scaleY/2, scaleZ/2)[1]
    time_out[ii] = t
    
    # Update fields for next step
    u_proj = project(u, W2)
    u_proj_old = project(u_old, W2)
    update_fields(u_proj, u_proj_old, v_old, a_old)
    w_old.vector()[:] = w.vector()
    
    # Update state variables
    A1_old.assign(local_project(A1, W3))
    
    # Print progress of calculation
    if ii%5==0:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Step: {}   |   Simulation Time: {} s  |     Iterations: {}".format(step, t, iter))
        print()
        

# Final step paraview output
file_results.write(u_v,t)
file_results.write(p_v, t)

# output final time histories
if t+dt>T_tot:
    b_out[ii] = b_app_max*(t-T_tot)/step2_time
else:
    b_out[ii] = 0.0
disp_out[ii] = w(scaleX/2, scaleY/2, scaleZ/2)[1]
time_out[ii] = t

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
ind2 = np.where(time_out==T_tot)[0][0]

expData = np.genfromtxt('Tan_B_step_7mT_data.csv', delimiter=',')

plt.figure()
plt.scatter(expData[:,0] - expData[0,0], expData[:,1], s=25,
                     edgecolors=(0.0, 0.0, 0.0,1),
                     color=(1, 1, 1, 1),
                     label='Experiment', linewidth=1.0)
plt.plot(time_out[0:ind]-T_tot, -(disp_out[0:ind]-disp_out[ind2]), linewidth=2.5, \
         color=colors[3], label='Simulation')
plt.axvline(0, c='k', linewidth=1.)
plt.axhline(0, c='k', linewidth=1.)
plt.axis('tight')
plt.xlabel(r"Time (s)")
plt.ylabel(r"$u_2^\mathrm{midspan}$ (mm)")
plt.grid(linestyle="--", linewidth=0.5, color='b')
plt.ylim(-0.0, 12)
plt.xlim(-0.0,step2_time)
plt.legend()

# save figure to file
fig = plt.gcf()
fig.set_size_inches(6, 4)
plt.tight_layout()
plt.savefig("plots/hard_magnetic_constant_bfield.png", dpi=600)


