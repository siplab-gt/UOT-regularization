function [S,L,M,R,diagnostic] = solver_RPCA_UOT_Beckman_ADMM(imsize,Y,Phi,lambda,gamma,kappa,mu,opts)
% solver_RPCA_UOT_Beckman_ADMM
% 
% This solves the RPCA with unbalanced optimal transport regularization
% problem via ADMM. To make the optimal transport regularization tractable,
% we utilized Beckman's formulation.
%
% [S,L,diagnostic] = solver_RPCA_UOT_Beckman_ADMM(A, b, lambda, rho, alpha)
% solves the following problem via ADMM:
%
%   min sum_t { 1/2*|| Phi(:,:,t)*(S(:,t)+L(:,t)) - Y(:,t) ||_2^2 }
%       + lambda*|| vec(S) ||_1
%       + gamma*|| L ||_*
%       + kappa*sum_{t=1}^{T-1} W_mu( S(:,t) , S(:,t+1) )
%   subject to
%       S,L >= 0
%       
% The solution is returned in the matrices S and L.
% 
% diagnostic is a structure that contains the primal and dual residual
% norms, and the rMSE at each iteration (if S_gt is provided in opts).
%
% Inputs:
% imsize            1x2 vector containing image size in <rows,columns>
% Y                 MxK matrix containing K measurement vectors as columns
% Phi               MxNxK tensor containing K measurement matrices in first
%                   two dimensions
% lambda            Sparsity parameter
% gamma             Low-rank parameter
% kappa             Temporal consistency parameter
% mu                Mass growth/decay parameter
% opts              struct contatining the following options:
%   .rho            augmented Lagrangian parameter
%   .maxiter        maximum ADMM iterations
%   .tolerance      stopping criteria for which both primal and dual
%                       residuals must reach
%   .beck_tau1      primal stepsize of proximal primal-dual algorithm
%   .beck_tau2      dual stepsize of proximal primal-dual algorithm
%   .beck_maxiter   maximum primal-dual algorithm iterations
%
% Copyright John Lee 2018.

% Default Parameters
rho = 0.5;
maxiter = 2000;
tolerance = 1e-3;
beckopts.tau1 = 0.1;
beckopts.tau2 = 1.0;
beckopts.maxiter = 1;

% Parameter via options (opts struct)
if exist('opts','var')
if isfield(opts,'rho'), rho = opts.rho; end
if isfield(opts,'maxiter'), maxiter = opts.maxiter; end
if isfield(opts,'tolerance'), tolerance = opts.tolerance; end
if isfield(opts,'beck_tau1'), beckopts.tau1 = opts.beck_tau1; end
if isfield(opts,'beck_tau2'), beckopts.tau2 = opts.beck_tau2; end
if isfield(opts,'beck_maxiter'), beckopts.maxiter = opts.beck_maxiter; end
end

% Initialization
[m,n,K] = size(Phi);
Div = GenerateDivergenceMatrices(imsize);
Phi_blk = mat2cell(Phi,size(Phi,1),size(Phi,2),ones(1,size(Phi,3))); Phi_blk = blkdiag(Phi_blk{:});
PhiTy = zeros(n,K); LL = zeros(m,m,K); UU = zeros(m,m,K);
for j = 1:K
    PhiTy(:,j) = Phi(:,:,j)'*Y(:,j);
    [LL(:,:,j),UU(:,:,j)] = factor(Phi(:,:,j),rho); 
end
f = ones(K,1)*3; f(1) = 2; f(end) = 2;

idx1 = 1:K-1;
idx2 = 2:K;
idxZ = 1:n;
idxW = n+1:2*n;

residual    = nan(maxiter,2);
S_rMSE      = nan(maxiter,1);
L_rMSE      = nan(maxiter,1);
objective   = nan(maxiter,1);

S = zeros(n,K); % primal variable
L = zeros(n,K); % primal variable
X = zeros(n,K); % auxiliary variable
T = zeros(n,K); % primal variable
ZW = zeros(2*n,K-1); % auxiliary variable
A = zeros(n,K); % dual variable
B = zeros(n,K-1); % dual variable
C = zeros(n,K-1); % dual variable
G = zeros(n,K); % dual variable
M = complex(zeros(n,K-1)); % primal variable
R = zeros(n,K-1); % auxiliary variable
D = zeros(n,K-1); % dual variable

% ADMM iterations
for k = 1:maxiter
    prevXTZW = [vec(X);vec(T);vec(ZW)];
    
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
    F = X - L + A;
    F(:,idx1) = F(:,idx1) + (ZW(idxZ,:) + B);
    F(:,idx2) = F(:,idx2) + (ZW(idxW,:) + C);
    F = F * diag(1./f);
    % S = Prox_NonNeg_l1(F,lambda/rho*ones(n,1)*(1./f)');
    S = Prox_NonNeg_l1(F, (lambda/rho*ones(n,1)*(1./f)') );
    
    % solve for L
    L = max(0,(X-S+A + T-G)/2);
    
    % solve for T
    T = Prox_NuclearNorm(L+G,gamma/rho);
    
    % solve for Z and W
    [ZW,M,R,D] = ...
        Prox_Beckman([S(:,idx1)-B;S(:,idx2)-C],mu,rho/kappa,Div,ZW,M,R,D,beckopts); % mu is multiplied by kappa
    
    % gradient ascent on dual
    A = A + (X-S-L);
    B = B + (ZW(idxZ,:)-S(:,idx1));
    C = C + (ZW(idxW,:)-S(:,idx2));
    G = G + (L-T);
    
    % compute residuals
    residual(k,1) = norm([vec(X-S-L);vec(L-T);vec(S(:,idx1)-ZW(idxZ,:));vec(S(:,idx2)-ZW(idxW,:))]);
    residual(k,2) = rho*norm([vec(X);vec(T);vec(ZW)]-prevXTZW);
    
    % compute rMSE (for testing)
    if exist('opts','var') 
        compute_rMSE = @(A,A_gt) norm(vec(A_gt-A))^2/norm(vec(A_gt))^2;
        if isfield(opts,'S_gt'), S_rMSE(k) = compute_rMSE(S,opts.S_gt); end
        if isfield(opts,'L_gt'), L_rMSE(k) = compute_rMSE(L,opts.L_gt); end
    end
    
    % compute objective
    objective(k) = 0.5 * sum_square(vec(Y)-Phi_blk*vec(S+L)) ...
                   + lambda*norm(vec(S),1) ...
                   + gamma*norm_nuc(L) ...
                   + kappa*(sum(abs(M(:)))+mu*norm(R(:),1));

    % termination criterion
    if residual(k,1) < tolerance && residual(k,2) < tolerance, break; end
    
%     % Display
%     if ~mod(k,10)
%     fontsize = 14;
%     figure(100);
%     subplot(521); display_batch(S,imsize); colorbar; title(['S, k=' num2str(k)]);
%     subplot(522); display_batch(T,imsize); colorbar; title('T');
%     subplot(523); display_batch(R,imsize); colorbar; title('R');
%     subplot(524); display_batch(X,imsize); colorbar; title('X');
%     subplot(5,2,5:6); plot(rMSE,'LineWidth',2); grid on; axis tight;
%     ylabel('rMSE','FontSize',fontsize); xlabel('Iteration','FontSize',fontsize);
%     subplot(5,2,7:8); semilogy(objective,'LineWidth',2); grid on; axis tight;
%     ylabel('Objective','FontSize',fontsize); xlabel('Iteration','FontSize',fontsize);
%     subplot(5,2,9:10); semilogy(residual,'LineWidth',2); grid on; axis tight;
%     ylabel('Residual Norm','FontSize',fontsize); xlabel('Iteration','FontSize',fontsize);
%     drawnow;
%     end
end

diagnostic.residual  = residual;%(1:k,:);
diagnostic.S_rMSE    = S_rMSE;%(1:k);
diagnostic.L_rMSE    = L_rMSE;%(1:k);
diagnostic.objective = objective;%(1:k);

% % Check equality constraint
% A_op = @(x) x(1:end/2,:) - x(end/2+1:end,:);
% K_op = @(m,x,r) real(conj(Div)*m) + A_op(x) - r;
% disp(['Equality constrain error = ' num2str( norm(K_op(M,ZW,R),'fro') )]);

end

function D = GenerateDivergenceMatrices(imsize)
Dx = speye(imsize(1)*imsize(2)) - circshift(speye(imsize(1)*imsize(2)),1);
Dx(1,imsize(1)*imsize(2)) = 0;
Dx(:,imsize(1):imsize(1):imsize(1)*imsize(2)) = zeros(imsize(1)*imsize(2),imsize(2));
Dy = speye(imsize(1)*imsize(2)) - circshift(speye(imsize(1)*imsize(2)),imsize(1));
Dy(:,end-imsize(1)+1:end) = zeros(imsize(1)*imsize(2),imsize(1));
D = Dx + 1i*Dy;
end

function [x,m,r,d] = Prox_Beckman(y,mu,rho,D_op,x,m,r,d,opts)
tau1    = opts.tau1;
tau2    = opts.tau2;
maxiter = opts.maxiter;
A_op = @(x) x(1:end/2,:) - x(end/2+1:end,:);
At_op = @(z) [z;-z];
K_op = @(m,x,r) real(conj(D_op)*m) + A_op(x) - r;
K_op_mxr = K_op(m,x,r);
for k = 1:maxiter
    prevm = m; prevx = x; prevr = r; prev_K_op_mxr = K_op_mxr;
    % Solve M (L2 norm shinkage)
    m = Prox_L21(prevm-tau1*conj(D_op')*d,tau1);
    % Solve x (Standard least-squares)
    % x = (rho*tau1)/(1+rho*tau1)*y + 1/(1+rho*tau1)*(prevx-tau1*At_op(d)); % Also works
    x = pos( (rho*tau1)/(1+rho*tau1)*y + 1/(1+rho*tau1)*(prevx-tau1*At_op(d)) ); % More proper
    % Solve r (L1 norm shinkage)
    r = Prox_L1(prevr+tau1*d, mu*tau1);
    % solve d (over-relaxation)
    K_op_mxr = K_op(m,x,r);
    d = d + tau2*( 2*K_op_mxr - prev_K_op_mxr );
end
end

function r = Prox_L1(r,rho)
r = sign(r).*max(0,abs(r)-rho);
end

function S = Prox_NonNeg_l1(A,rho)
S = max(0,A-rho);
end

function L = Prox_NuclearNorm(A,rho)
prox_l1 = @(x) max(0,x-rho) - max(0,-x-rho);
[U,S,V] = svd(A,'econ');
L = U*diag(prox_l1(diag(S)))*V';
end

function m = Prox_L21(m,rho)
abs_m = abs(m);
m = (1 - rho./abs_m).*m;
m(abs_m<rho) = 0;
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