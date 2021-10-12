function [S,L,diagnostic] = solver_RPCA_ADMM(Y,Phi,lambda,gamma,opts)

% Default Parameters
rho = 0.5;
maxiter = 2000;
tolerance = 1e-3;

% Parameter via options (opts struct)
if exist('opts','var')
if isfield(opts,'rho'), rho = opts.rho; end
if isfield(opts,'maxiter'), maxiter = opts.maxiter; end
if isfield(opts,'tolerance'), tolerance = opts.tolerance; end
end

% Initialization
[m,n,K] = size(Phi);
Phi_blk = mat2cell(Phi,size(Phi,1),size(Phi,2),ones(1,size(Phi,3))); Phi_blk = blkdiag(Phi_blk{:});
PhiTy = zeros(n,K); LL = zeros(m,m,K); UU = zeros(m,m,K);
for j = 1:K
    PhiTy(:,j) = Phi(:,:,j)'*Y(:,j);
    [LL(:,:,j),UU(:,:,j)] = factor(Phi(:,:,j),rho); 
end

S = zeros(n,K); % primal variable
L = zeros(n,K); % primal variable
X = zeros(n,K); % auxiliary variable
T = zeros(n,K); % auxiliary variable
A = zeros(n,K); % auxiliary variable
G = zeros(n,K); % auxiliary variable

residual    = nan(maxiter,2);
objective   = nan(maxiter,1);

% ADMM iterations
for k = 1:maxiter
    prevXT = [vec(X);vec(T)];
    
    % solve for X
    Q = PhiTy + rho*(S+L-A);
    for j = 1:K
    if( size(Phi,1) >= size(Phi,2) )
       X(:,j) = UU(:,:,j) \ (LL(:,:,j) \ Q(:,j));
    else
       X(:,j) = Q(:,j)/rho - (Phi(:,:,j)'*(UU(:,:,j) \ ( LL(:,:,j) \ (Phi(:,:,j)*Q(:,j)) )))/rho^2;
    end
    end
    
    % solve for S
    S = Prox_NonNeg_l1(X-L+A,lambda/rho);
    
    % solve for L
    L = max(0,(X-S+A + T-G)/2);
    
    % solve for T
    T = Prox_NuclearNorm(L+G,gamma/rho);
    
    % gradient ascent on dual
    A = A + (X-S-L);
    G = G + (L-T);
    
    % compute residuals
    residual(k,1) = norm([vec(X-S-L);vec(L-T)]);
    residual(k,2) = rho*norm([vec(X);vec(T)]-prevXT);

    % compute objective
    objective(k) = 0.5 * sum_square(vec(Y)-Phi_blk*vec(S+L)) ...
                   + lambda*norm(vec(S),1) ...
                   + gamma*norm_nuc(L);
    
    % termination criterion
    if residual(k,1) < tolerance && residual(k,2) < tolerance, break; end
end

diagnostic.residual  = residual(1:k,:);
diagnostic.objective = objective(1:k,:);

end

function S = Prox_NonNeg_l1(A,rho)
S = max(0,A-rho);
end

function L = Prox_NuclearNorm(A,rho)
prox_l1 = @(x) max(0,x-rho) - max(0,-x-rho);
[U,S,V] = svd(A,'econ');
L = U*diag(prox_l1(diag(S)))*V';
end

function [L,U] = factor(Phi, rho)
% Taken from Stephen Boyd's code:
% https://web.stanford.edu/~boyd/papers/admm/lasso/lasso.html
[m, n] = size(Phi);
if ( m >= n )
   L = chol( Phi'*Phi + rho*speye(n), 'lower' );
else
   L = chol( speye(m) + 1/rho*(Phi*Phi'), 'lower' );
end
U = L';
end