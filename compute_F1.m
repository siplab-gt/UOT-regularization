function val = compute_F1(A,B,th)
A = vec(A>=th); B = vec(B>=th);
val = 2*sum(vec((A>0)&(B>0))) / ( sum(vec(A>0)) + sum(vec(B>0)) );
end