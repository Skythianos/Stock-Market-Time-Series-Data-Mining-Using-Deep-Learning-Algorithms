# -*- coding: utf-8 -*-
#   MatLab
#
#   load('s.mat');
#   G = cov(s);
#   [T, N] = size(s);
#   A = zeros(N, N);
#   for t=2:1:T
#       A = A + pinv(s(t-1,:).'*s(t-1,:))*(s(t-1,:).'*s(t,:));
#   end
#   pinvg = pinv(G)^0.5;
#   [v1, l1] = maxeig(pinvg.'*A*G*A.'*pinvg);
#   w = real(pinvg*v1);

import csv
import numpy as np
from numpy.core.multiarray import zeros
from scipy import linalg
import time

start_time = time.clock()

Theano = False

porfolio=1000
fee=1
reader = csv.reader(open("C:\s980.csv", "rb"), delimiter=',')
S = np.matrix(list(reader)).astype('double')
[T, N] = S.shape
G = np.cov(S, rowvar=False)

A = zeros((N, N))
for t in range(1, T): # sum 2 to T
    STMinusOneTranspose = S[t-1, :].T
    A = A + (linalg.pinv2((S[t-1,:].T).dot(S[t-1,:]))).dot((S[t-1,:].T).dot(S[t,:]));       
pingv = linalg.sqrtm(linalg.pinv2(G))
X = (pingv.T).dot(A).dot(G).dot((A.T)).dot(pingv)
[maxEigenValue, maxEigenVector] = linalg.eigh(X, eigvals_only=False, eigvals=(N-1, N-1))
w = pingv.real.dot(maxEigenVector.real)

if w.dot(A[:,N-1])>pandas.stats.moments.ewma(A[:,N-1], span=10):
	portfolio=porfolio+porftolio*w.dot(A[:,N-1])-fee
else
	portfolio=porfolio+porftolio*-w.dot(A[:,N-1])-fee
end_time = time.clock()
print end_time-start_time