import scipy.linalg as la
import numpy as np

def statespace(xi, K, omega_n, dt, acceleration):
   g = 9.81
   disp = np.zeros((len(omega_n),1)) 
   for j in range(len(omega_n)):
     wn = omega_n[j,0]
     A = np.array([[0, 1],[-wn**2, -2*xi*wn]])
     D, V = la.eig(A)
     ep = np.array([[np.exp(D[0]*dt), 0],[0, np.exp(D[1]*dt)]])
     Ad = V.dot(ep).dot(np.linalg.inv(V))
     Bd = np.linalg.inv(A).dot(Ad-np.eye((len(A))))
     z = [[0],[0]]
     d = np.zeros((len(acceleration),1))
     v = np.zeros((len(acceleration),1))
     for i in range(len(acceleration)):
        z = np.real(Ad).dot(z) + np.real(Bd).dot([[0],[-acceleration[i] * g]])
        d[i] = z[0, 0]
        v[i] = z[1, 0]
     disp[j] = max(abs(d))
   return disp