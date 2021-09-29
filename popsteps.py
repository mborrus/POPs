#Python 3 code to calculate POPs according to Sheshadri and Plumb, JAS (2017)
#Author: Alison Ming, license: cc-non-commercial by attribution (2020)
#Adapted from Matlab code from Aditi Sheshadri.

import numpy as np
from numpy.lib.stride_tricks import as_strided

def _check_arg(x, xname):
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError('%s must be one-dimensional.' % xname)
    return x

def crosscorrelation(x, y, maxlag):
    """
    Cross correlation with a maximum number of lags.

    `x` and `y` must be one-dimensional numpy arrays with the same length.

    This computes the same result as
        numpy.correlate(x, y, mode='full')[len(a)-maxlag-1:len(a)+maxlag]

    The return vaue has length 2*maxlag + 1.
    """
    x = _check_arg(x, 'x')
    y = _check_arg(y, 'y')
    py = np.pad(y.conj(), 2*maxlag, mode='constant')
    T = as_strided(py[2*maxlag:], shape=(2*maxlag+1, len(y) + 2*maxlag),
                   strides=(-py.strides[0], py.strides[0]))
    px = np.pad(x, maxlag, mode='constant')
    return T.dot(px)

import scipy.io
import pygeode as pyg

# !!! Change the u, p and lat below to the actually fields you want to use !!!
# This assumes that the seasonal cycle has been removed from u and it has been detrended.
u = np.ones(1000,20,40) # This is a zonal mean zonal wind on (1000 days, 20 latitudes and 40 pressure levels)
p = np.ones(40,)        # pressure levels in hPa
lat = np.ones(20,)      # latitudes in radians

u_shape = np.shape(u)

#Make pressure-lat weighting
#pres_weighting = sqrt(dp * cos lat)
dp = np.tile(np.gradient(p), [u_shape[0], u_shape[1], 1])
cosphi = np.transpose(np.tile(np.cos(lat), [u_shape[0], u_shape[2], 1]), [0, 2, 1])

pres_weighting =  np.sqrt(dp * 100 * cosphi)
du_weighted_in = u * pres_weighting

du_shape = np.shape(du_weighted_in)

#EOF calculation

#Calculate SVD. du = U S V^T
u_q, s_q, vh_q = np.linalg.svd(np.reshape(du_weighted_in[:],[du_shape[0], du_shape[1]*du_shape[2]]), full_matrices=False)

#Note that the EOFs are vh_q transpose with the pressure weighting removed.
#EOFs = vh_q.T / pres_weighting
eof_order_length = du_shape[1] * du_shape[2]

#reshape pressure weighting to work for unweighting EOFs
pres_weighting_eofs = np.transpose(pres_weighting[:eof_order_length,:,:], [1, 2, 0])
eof_q = np.reshape(vh_q.T, [du_shape[1], du_shape[2], du_shape[0]])/pres_weighting_eofs

#Calculate PCs
us_q = u_q * s_q

#fraction of variance explained by EOFs
print('Fraction of variance explained by EOFs 1 and 2: ', ((s_q.squeeze()[:])**2/np.sum(du_weighted_in.squeeze()[:]**2))[:2]*100)

#POP calculation

#Choose lag of 20 days. This depends on the lag at which the cross correlation between PC1 and PC2 maximises.
lag = 20

c = np.zeros([len(s_q),len(s_q)])
b = np.zeros([len(s_q),len(s_q), 2*lag+1])
A_at_lag_0 = np.zeros([len(s_q),])

#find lag 0 autocorrelation for scaling
for i in range(len(s_q)):
    A_at_lag_0[i] = crosscorrelation(us_q[:,i], us_q[:,i], 0)
    
# A (lag tau) = lag covariance of PC (lag tau) and PC (lag 0). Equation (5) in paper.
# C = S A S^-1
for i in range(len(s_q)):
    for j in range(len(s_q)):
        A = crosscorrelation(us_q[:,i], us_q[:,j], lag)
        #Scale the crosscorrelation so that the autocorrelation at lag 0 = 1
        b[i,j,:] = A/np.sqrt(A_at_lag_0[i] * A_at_lag_0[j])
        #Scale lag covariance 
        c[i,j] = (s_q[i]/s_q[j])*(b[i,j,-1])

# Eigen analysis of c
eigenValues, eigenVectors = np.linalg.eig(c)

# Sort the eigenvalues of c (by real value)
idx = eigenValues.argsort()[::-1]   
ds = eigenValues[idx]
vs = eigenVectors[:,idx]

# Pops are EOFs * vs. Equation (7) in paper.
#Note: You need to divide by the pressure weighting to get the actual POPS
pop_q1 = np.matmul(vh_q.T, vs) 
pop_q = np.reshape(pop_q1, [du_shape[1], du_shape[2], du_shape[0]])/ pres_weighting_eofs
#IDENTIFY USEFUL POPs

#This gives all the POPs. Now we need to identify the useful ones. This is done by selecting those POPs onto which the first two EOFs project the most strongly

inverse_vs = np.linalg.inv(vs)

dl = np.zeros(np.shape(inverse_vs), dtype=complex)

for i in range(np.shape(inverse_vs)[0]):
    dl[:,i] = inverse_vs[:,i]/np.sum(np.abs(inverse_vs[:,i]))
    
yy = np.max(np.abs(dl),0)
ii = np.argmax(np.abs(dl),0)

print('Indices of the first few useful POPs: ', ii[:5])
# Equation (9)
# 1/ \lambda_R
print('Decay time scale of first useful POP: ', 1/np.real(-np.log(ds[ii[0]])/lag))
# 2 pi 1/lambda_I
print('Oscillation time scale of first useful POP: ', 1/np.imag(-np.log(ds[ii[0]])/lag)*2*np.pi)
