tic;
load('s.mat');
G = cov(s);
A = est_A(s);
w = mr_opt_w(A, G);
toc