function w = mr_opt_w(A, G)
    pinvg = pinv(G)^0.5;
    [v1, l1] = maxeig(pinvg.'*A*G*A.'*pinvg);
    w = real(pinvg*v1);
end