function [S,L,diagnostic] = solver_RPCA_ConvL1DF_ADMM(imsize,Y,Phi,lambda,gamma,kappa,opts)

% Default Parameters
rho = 0.5;
maxiter = 2000;
tolerance = 1e-3;
sigma = 1.5;

% Parameter via options (opts struct)
if exist('opts','var')
if isfield(opts,'rho'), rho = opts.rho; end
if isfield(opts,'maxiter'), maxiter = opts.maxiter; end
if isfield(opts,'tolerance'), tolerance = opts.tolerance; end
if isfield(opts,'sigma'), sigma = opts.sigma; end
end

% Initialization
[m,n,K] = size(Phi);
Phi_blk = mat2cell(Phi,size(Phi,1),size(Phi,2),ones(1,size(Phi,3))); Phi_blk = blkdiag(Phi_blk{:});
PhiTy = zeros(n,K); LL = zeros(m,m,K); UU = zeros(m,m,K);
for j = 1:K
    PhiTy(:,j) = Phi(:,:,j)'*Y(:,j);
    [LL(:,:,j),UU(:,:,j)] = factor(Phi(:,:,j),rho); 
end
PSF = fspecial('gaussian',[3,3],sigma);
F = convmtx2(PSF,imsize);
Mask = zeros(imsize+2); Mask(1:end,1) = 1; Mask(1:end,end) = 1; Mask(1,1:end)= 1; Mask(end,1:end)=1;
F(find(Mask),:) = [];
FtF = F'*F;
[L1,U1] = factor(  F,2);
[L2,U2] = factor(2*F,2);

idx1 = 1:K-1;
idx2 = 2:K;
idxZ = 1:n;
idxW = n+1:2*n;

residual    = nan(maxiter,2);
rMSE        = nan(maxiter,1);
objective   = nan(maxiter,1);

S = zeros(n,K); % primal variable
L = zeros(n,K); % primal variable
X = zeros(n,K); % auxiliary variable
T = zeros(n,K); % primal variable
R = zeros(n,K); % primal variable
ZW = zeros(2*n,K-1); % auxiliary variable
A = zeros(n,K); % dual variable
B = zeros(n,K-1); % dual variable
C = zeros(n,K-1); % dual variable
G = zeros(n,K); % dual variable
H = zeros(n,K); % dual variable

% ADMM iterations
for k = 1:maxiter
    prevXTRZW = [vec(X);vec(T);vec(R);vec(ZW)];
    
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
    S = (R-H) + (X-L+A);
    S(:,idx1) = S(:,idx1) + Ft_op(imsize,(ZW(idxZ,:) + B),PSF);
    S(:,idx2) = S(:,idx2) + Ft_op(imsize,(ZW(idxW,:) + C),PSF);
    S(:,[1,K]) = U1 \ (L1 \ S(:,[1,K]));
    S(:,2:K-1) = U2 \ (L2 \ S(:,2:K-1));
    
    % solve for R
    R = Prox_NonNeg_l1(S+H,lambda/rho);
    
    % solve for L
    L = max(0,(X-S+A + T-G)/2);
    
    % solve for T
    T = Prox_NuclearNorm(L+G,gamma/rho);
    
    % solve for Z
    ZW(idxZ,:) = ZW(idxW,:) + Prox_L1(F_op(imsize,S(:,idx1),PSF)-ZW(idxW,:)-B,kappa/rho);
    
    % solve for W
    ZW(idxW,:) = ZW(idxZ,:) + Prox_L1(F_op(imsize,S(:,idx2),PSF)-ZW(idxZ,:)-C,kappa/rho);
    
    % gradient ascent on dual
    A = A + (X-S-L);
    B = B + (ZW(idxZ,:)-F_op(imsize,S(:,idx1),PSF));
    C = C + (ZW(idxW,:)-F_op(imsize,S(:,idx2),PSF));
    G = G + (L-T);
    H = H + (S-R);
    
    % compute residuals
    residual(k,1) = norm([vec(X-S-L);vec(L-T);vec(S-R);vec(F_op(imsize,S(:,idx1),PSF)-ZW(idxZ,:));vec(F_op(imsize,S(:,idx2),PSF)-ZW(idxW,:))]);
    residual(k,2) = rho*norm([vec(X);vec(T);vec(R);vec(ZW)]-prevXTRZW);
    
    % compute rMSE (for testing)
    if exist('opts','var') && isfield(opts,'S_gt'), rMSE(k) = norm(vec(opts.S_gt-S))^2/norm(vec(opts.S_gt))^2; end
    
    % compute objective
    objective(k) = 0.5 * sum_square(vec(Y)-Phi_blk*vec(S+L)) ...
                   + lambda*norm(vec(S),1) ...
                   + gamma*norm_nuc(L) ...
                   + kappa*norm(vec(F_op(imsize,S(:,1:end-1),PSF)-F_op(imsize,S(:,2:end),PSF)),1);

    % termination criterion
    if residual(k,1) < tolerance && residual(k,2) < tolerance, break; end
end

diagnostic.residual  = residual(1:k,:);
diagnostic.rMSE      = rMSE(1:k);
diagnostic.objective = objective(1:k);

end

function FX = F_op(imsize,X,PSF)
f_op  = @(x) vec(conv2( reshape(x,imsize) , PSF , 'same' ));
FX = zeros(size(X));
for i = 1:size(X,2)
    FX(:,i) = f_op(X(:,i));
end
end

function FtX = Ft_op(imsize,X,PSF)
ft_op = @(x) vec(conv2( reshape(x,imsize) , rot90(PSF,2) , 'same' ));
FtX = zeros(size(X));
for i = 1:size(X,2)
    FtX(:,i) = ft_op(X(:,i));
end
end

function S = Prox_NonNeg_l1(A,rho)
S = max(0,A-rho);
end

function L = Prox_NuclearNorm(A,rho)
prox_l1 = @(x) max(0,x-rho) - max(0,-x-rho);
[U,S,V] = svd(A,'econ');
L = U*diag(prox_l1(diag(S)))*V';
end

function r = Prox_L1(r,rho)
r = sign(r).*max(0,abs(r)-rho);
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