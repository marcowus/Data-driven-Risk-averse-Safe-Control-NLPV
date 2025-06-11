import time
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse
from scipy.linalg import sqrtm

tic = time.time()

# Time sequence
T0 = 0
Ts = 0.4
Tf = 20
t = np.arange(T0, Tf+Ts, Ts)

nSteps = t.size

# System matrices
M = 1
k_e = 0.33

A_1 = np.array([[1, Ts], [-Ts*k_e/M, (M-0.6*Ts)/M]])
A_2 = np.array([[1, Ts], [-Ts*k_e/M, (M-1.6*Ts)/M]])

B = np.array([[0], [Ts/M]])

nStates = A_1.shape[0]
nInputs = B.shape[1]

nPolytops = 2
vertex_val = 18

F = np.array([[1/vertex_val, 0], [-1/vertex_val, 0], [0, 1/vertex_val], [0, -1/vertex_val]])
g = np.array([1, 1, 1, 1])

F_vertices = np.array([[vertex_val, vertex_val, -vertex_val, -vertex_val],
                       [vertex_val, -vertex_val, -vertex_val, vertex_val]])

# Controller parameters
lambda_val = 0.8

Q = np.diag([18*Ts*k_e/M, 0])

epsilon_1 = 0.01
epsilon_2 = 0.01

tau_1 = 0.02
tau_2 = 0.03

eta_1 = 0.01
eta_2 = 0.01

delta = 0.001
delta_n = nStates + 2*np.sqrt(nStates*np.log(1/delta)) + 2*np.log(1/delta)

# Noise generation
varNoise = 0.05
coVarNoise = 0
coVarMatrix = np.array([[varNoise, coVarNoise], [coVarNoise, varNoise]])
nRealization = 100*nStates
nSample = 100000
noise = np.sqrt(varNoise) * np.random.randn(nRealization, nSample)
w = np.zeros((nRealization, nSample))
for jj in range(nRealization // nStates):
    w[2*jj:2*jj+2, :] = noise[2*jj:2*jj+2, :]

# Data collection
N = nStates + 8 

X = np.zeros((nStates, N+1))
X[:, 0] = 0.04*np.random.rand(nStates) - 1

U0 = 5 * np.random.randn(nInputs, N)

V = np.zeros(N)
W_1 = np.zeros(N)
W_2 = np.zeros(N)

OMEGA_0 = np.zeros((nPolytops, N))

W0 = noise[0:N, 0:nStates].T
W0[0, :] = 0

for kk in range(N):
    V[kk] = 1.1 + 0.5*np.sin(0.03*kk)
    W_1[kk] = V[kk] - 0.6
    W_2[kk] = 1 - W_1[kk]
    OMEGA_0[:, kk] = [W_1[kk], W_2[kk]]
    X[0, kk+1] = X[0, kk] + Ts*X[1, kk]
    X[1, kk+1] = X[1, kk] - Ts*(k_e/M)*np.exp(-X[0, kk])*X[0, kk] - (Ts/M)*V[kk]*X[1, kk] + (Ts/M)*U0[0, kk] + W0[1, kk]

X0 = X[:, 0:N]
X1 = X[:, 1:N+1]
Uw = np.zeros((nInputs*nPolytops, N))
for kk in range(N):
    Uw[:, kk] = np.kron(OMEGA_0[:, kk], U0[:, kk])
Xw = np.zeros((nStates*nPolytops, N))
for kk in range(N):
    Xw[:, kk] = np.kron(OMEGA_0[:, kk], X0[:, kk])
Z0 = np.vstack([Xw, np.exp(-X0[0, :])*X0[0, :]])

# print(np.linalg.matrix_rank(Z0))
print(Z0)

if np.linalg.matrix_rank(Z0) < min(Z0.shape):
    raise ValueError('Data are not full row rank!')

nZ = Z0.shape[0]

# Optimization
d_1 = F_vertices[:, 3]
d_2 = F_vertices[:, 3]

gamma = cp.Variable(nonneg=True)

kappa_1 = cp.Variable()
kappa_2 = cp.Variable()

phi_1 = cp.Variable()
phi_2 = cp.Variable()

v_1 = cp.Variable(nonneg=True)
v_2 = cp.Variable(nonneg=True)

P_1 = cp.Variable((nStates, nStates), PSD=True)
P_2 = cp.Variable((nStates, nStates), PSD=True)

Y_1 = cp.Variable((N, nStates))
Y_2 = cp.Variable((N, nStates))

M_1 = cp.Variable((N, N), PSD=True)
M_2 = cp.Variable((N, N), PSD=True)

G_K_nl = cp.Variable((N, nZ-nStates*nPolytops))

constraints = []

constraints += [Z0 @ Y_1 == cp.vstack([P_1 , np.zeros((nZ-nStates, nStates))])]
constraints += [Z0 @ Y_2 == cp.vstack([P_2, np.zeros((nZ-nStates, nStates))])]
constraints += [Z0 @ G_K_nl == cp.vstack([np.zeros((nStates*nPolytops, nZ-nStates*nPolytops)),
                                         np.eye(nZ-nStates*nPolytops)])]
sqrtCoVar = np.sqrt(varNoise)*np.eye(nStates)
block4 = cp.bmat([[P_1 - ((1+1/tau_1)/(lambda_val-eta_1))*np.eye(nStates), X1 @ Y_1, kappa_1*sqrtCoVar],
                  [(X1 @ Y_1).T, ((lambda_val-eta_1)/(1+tau_1))*P_1, np.zeros((nStates, nStates))],
                  [(kappa_1*sqrtCoVar).T, np.zeros((nStates, nStates)), (eta_1/delta_n)*np.eye(nStates)]])
block5 = cp.bmat([[P_2 - ((1+1/tau_2)/(lambda_val-eta_2))*np.eye(nStates), X1 @ Y_2, kappa_2*sqrtCoVar],
                  [(X1 @ Y_2).T, ((lambda_val-eta_2)/(1+tau_2))*P_2, np.zeros((nStates, nStates))],
                  [(kappa_2*sqrtCoVar).T, np.zeros((nStates, nStates)), (eta_2/delta_n)*np.eye(nStates)]])
constraints += [block4 >> 0, block5 >> 0]

constraints += [cp.bmat([[epsilon_1*np.eye(nStates), np.eye(nStates)], [np.eye(nStates), P_1]]) >> 0]
constraints += [cp.bmat([[epsilon_2*np.eye(nStates), np.eye(nStates)], [np.eye(nStates), P_2]]) >> 0]

constraints += [cp.bmat([[np.eye(nStates), gamma*Q], [gamma*Q.T, P_1]]) >> 0]
constraints += [cp.bmat([[np.eye(nStates), gamma*Q], [gamma*Q.T, P_2]]) >> 0]

constraints += [cp.bmat([[gamma*np.eye(nStates), X1 @ G_K_nl], [G_K_nl.T @ X1.T, gamma*np.eye(nZ-nStates*nPolytops)]]) >> 0]

constraints += [cp.bmat([[M_1, Y_1], [Y_1.T, P_1]]) >> 0]
constraints += [cp.bmat([[M_2, Y_2], [Y_2.T, P_2]]) >> 0]

constraints += [cp.bmat([[phi_1+((2+1/tau_1)/(1+tau_1)), kappa_1/np.sqrt(1+tau_1)], [kappa_1/np.sqrt(1+tau_1), 1]]) >> 0]
constraints += [cp.bmat([[phi_2+((2+1/tau_2)/(1+tau_2)), kappa_2/np.sqrt(1+tau_2)], [kappa_2/np.sqrt(1+tau_2), 1]]) >> 0]

constraints += [cp.trace(M_1) <= phi_1, cp.trace(M_2) <= phi_2, gamma >= 0]

# for i in range(4):
#    constraints += [cp.bmat([[P_1, P_1 @ F[i, :].reshape(-1, 1)], [(P_1 @ F[i, :].reshape(-1, 1)).T, g[i]**2]]) >> 0]
#    constraints += [cp.bmat([[P_2, P_2 @ F[i, :].reshape(-1, 1)], [(P_2 @ F[i, :].reshape(-1, 1)).T, g[i]**2]]) >> 0]
for i in range(4):
    fi = cp.reshape(F[i, :], (nStates, 1))
    g_scalar = cp.Constant([[g[i] ** 2]])
    Pf_1 = P_1 @ fi
    Pf_2 = P_2 @ fi
    constraints += [cp.bmat([[P_1, Pf_1], [Pf_1.T, g_scalar]]) >> 0]
    constraints += [cp.bmat([[P_2, Pf_2], [Pf_2.T, g_scalar]]) >> 0]

constraints += [cp.bmat([[cp.Constant([[1]]), v_1*d_1.reshape(1, -1)], [v_1*d_1.reshape(-1, 1), P_1]]) >> 0]
constraints += [cp.bmat([[cp.Constant([[1]]), v_2*d_2.reshape(1, -1)], [v_2*d_2.reshape(-1, 1), P_2]]) >> 0]

prob = cp.Problem(cp.Minimize(gamma + (kappa_1+kappa_2) + (phi_1+phi_2) - 1000*(v_1+v_2)), constraints)
prob.solve(solver=cp.SCS, verbose=False)

gamma_opt = gamma.value
P_1_opt = P_1.value
P_2_opt = P_2.value
Y_1_opt = Y_1.value
Y_2_opt = Y_2.value
G_K_nl_opt = G_K_nl.value
K_opt = U0 @ np.hstack([Y_1_opt @ np.linalg.inv(P_1_opt), Y_2_opt @ np.linalg.inv(P_2_opt), G_K_nl_opt])

# Simulation
x0 = np.array([12, -17.4])
x = np.zeros((nStates*(nRealization//nStates), nSteps+1))
for ii in range(nRealization//nStates):
    x[2*ii:2*ii+2, 0] = x0
u = np.zeros((nInputs, nSteps))
for ii in range(nRealization//nStates):
    for k in range(nSteps):
        v_k = 1.1 + 0.5*np.sin(0.03*(k+1))
        w_1 = v_k - 0.6
        w_2 = 1 - w_1
        u[0, k] = K_opt @ np.concatenate([np.kron(x[2*ii:2*ii+2, k], [w_1, w_2]), [np.exp(-x[2*ii, k])*x[2*ii, k]]])
        x[2*ii, k+1] = x[2*ii, k] + Ts*x[2*ii+1, k]
        x[2*ii+1, k+1] = x[2*ii+1, k] - Ts*(k_e/M)*np.exp(-x[2*ii, k])*x[2*ii, k] - (Ts/M)*v_k*x[2*ii+1, k] + (Ts/M)*u[0, k] + w[2*ii, k]

toc = time.time() - tic

# Plot results
plt.figure(1)
# ax = plt.gca()
plt.plot(t, u.T, 'b-', linewidth=2)
plt.xlim([0,t[-1]])
plt.grid(True)
plt.xlabel('Time [s]', fontsize=15)
plt.ylabel('u(k)', fontsize=15)
plt.suptitle("Control input")

# Plot 2: States and Phase plot
fig2 = plt.figure(2)
gs = fig2.add_gridspec(2, 2)

# x1 over time
ax1 = fig2.add_subplot(gs[0, 0])
ax1.plot(t, x[0, :-1], 'b-', linewidth=2)
ax1.set_xlim([0,t[-1]])
ax1.grid(True)
ax1.set_xlabel('Time [s]', fontsize=15)
ax1.set_ylabel(r'$x_1$ [m]', fontsize=15)

# x2 over time
ax2 = fig2.add_subplot(gs[1, 0])
ax2.plot(t, x[1, :-1], 'b-', linewidth=2)
ax2.set_xlim([0,t[-1]])
ax2.grid(True)
ax2.set_xlabel('Time [s]', fontsize=15)
ax2.set_ylabel(r'$x_2$ [m/s]', fontsize=15)

# Phase plot
ax3 = fig2.add_subplot(gs[:, 1])
ax3.plot(x[0, :-1], x[1, :-1], 'r-', linewidth=2)
ax3.set_xlim([0,t[-1]])
ax3.grid(True)
ax3.set_xlabel(r'$x_1 [m]$', fontsize=15)
ax3.set_ylabel(r'$x_2 [m/s]$', fontsize=15)

fig2.suptitle(f'Objective value: gamma = {gamma_opt:.5f}')
plt.tight_layout()

nLS = 8

fig3, ax3 = plt.subplots()
ax3.set_aspect('equal')
ax3.grid(True)

greenCol = [0.4660, 0.6740, 0.1880]
redCol = np.array([237, 28, 36]) / 256

polygon = Polygon(F_vertices.T, closed=True, edgecolor=greenCol, facecolor='none', linewidth=1)
ax3.add_patch(polygon)

for k in range(1, nLS + 1):
    scaled_vertices = (lambda_val ** (k - 1)) * F_vertices
    polygon_k = Polygon(scaled_vertices.T, closed=True,
                        edgecolor='green', linestyle='--' if k > 1 else '-', facecolor='none')
    ax3.add_patch(polygon_k)

ax3.plot(x0[0], x0[1], 'sm', markersize=15, markerfacecolor='m', label='Initial value')
ax3.plot(x[0, -1], x[1, -1], 'pm', markersize=15, markerfacecolor='m', label='Desired value')

for ii in range(nRealization // nStates):
    ax3.plot(x[2 * ii, :], x[2 * ii + 1, :], 'r.-', linewidth=1, markersize=6)

ax3.set_xlabel(r'$x_1$ [m]', fontsize=15)
ax3.set_ylabel(r'$x_2$ [m/s]', fontsize=15)
ax3.legend()

fig3.suptitle("Phase plot")

plt.show()