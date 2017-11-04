function A = est_A(s)
    [T, N] = size(s);
    A = zeros(N, N);
    for t=2:1:T
        A = A + pinv(s(t-1,:).'*s(t-1,:))*(s(t-1,:).'*s(t,:));
        %A = A + pinv(s(t-1,:).'*s(t-1,:));
    end
end