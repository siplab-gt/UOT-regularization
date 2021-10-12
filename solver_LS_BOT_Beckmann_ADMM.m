function [s,m,diagnostic] = solver_LS_BOT_Beckmann_ADMM(imsize,y,Phi,x0,kappa,opts)
% solver_LS_BOT_Beckman_ADMM
% 
% This solves least-squares problem with balanced optimal transport
% regularization via ADMM. To make the optimal transport regularization 
% tractable, we utilize Beckmann's formulation.
%
%   min  1/2 * || y - Phi*s ||_2^2 + kappa  * W( s , s0 ) )
%   s.t. s >= 0
%       
% diagnostic is a structure that contains the primal and dual residual
% norms, and the rMSE at each iteration (if S_gt is provided in opts).
%
% Inputs:
% imsize            1x2 vector containing image size in <rows,columns>
% y                 M vector containing measurements
% Phi               MxN matrix containing measurement matrix
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
% Copyright John Lee 2020.

% Default Parameters
rho = 0.5;
maxiter = 200;
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
[~,n] = size(Phi);
Div = GenerateDivergenceMatrices(imsize);
PhiTy = Phi'*y;
[LL,UU] = factor(Phi,rho); 

residual    = nan(maxiter,2);
s_rMSE      = nan(maxiter,1);
objective   = nan(maxiter,1);

s = zeros(n,1); % primal variable
x = zeros(n,1); % auxiliary variable
z = zeros(n,1); % auxiliary variable
a = zeros(n,1); % dual variable
b = zeros(n,1); % dual variable
m = complex(zeros(n,1)); % primal variable
d = zeros(n,1); % dual variable

% ADMM iterations
for k = 1:maxiter
    prevxz = [vec(x);vec(z)];
    
    % solve for x (least squares term)
    q = PhiTy + rho*(s-a);
    if( size(Phi,1) >= size(Phi,2) )
       x = UU \ (LL \ q);
    else
       x = q/rho - (Phi'*(UU \ ( LL \ (Phi*q) )))/rho^2;
    end
    
    % solve for s (non-negative constraints)
    s = pos(x + a + z + b)/2;
    
    % solve for z (Unbalanced OT regularization)
    [z,m,d] = ...
        Prox_Beckman(s-b,x0,rho/kappa,Div,z,m,d,beckopts); % mu is multiplied by kappa
    
    % gradient ascent on dual
    a = a + (x-s);
    b = b + (z-s);
    
    % compute residuals
    residual(k,1) = norm([vec(x-s);vec(z-s)]);
    residual(k,2) = rho*norm([vec(x);vec(z)]-prevxz);
    
    % compute rMSE (for testing)
    if exist('opts','var') 
        compute_rMSE = @(A,A_gt) norm(vec(A_gt-A))^2/norm(vec(A_gt))^2;
        if isfield(opts,'S_gt'), s_rMSE(k) = compute_rMSE(s,opts.S_gt); end
    end
    
    % compute objective
    objective(k) = 0.5 * sum_square(vec(y)-Phi*s) ...
                   + kappa*(sum(abs(m)));

    % termination criterion
    if residual(k,1) < tolerance && residual(k,2) < tolerance, break; end
    
%     % Display
%     if ~mod(k,10)
%     fontsize = 14;
%     figure(200);
%     subplot(331); imagesc(reshape(s,imsize)); colorbar; title(['s, k=' num2str(k)]);
%     subplot(332); imagesc(reshape(r,imsize)); colorbar; title('r');
%     subplot(333); imagesc(reshape(x,imsize)); colorbar; title('x');
%     subplot(3,3,4:6); semilogy(objective,'LineWidth',2); grid on; axis tight;
%     ylabel('Objective','FontSize',fontsize); xlabel('Iteration','FontSize',fontsize);
%     subplot(3,3,7:9); semilogy(residual,'LineWidth',2); grid on; axis tight;
%     ylabel('Residual Norm','FontSize',fontsize); xlabel('Iteration','FontSize',fontsize);
%     drawnow;
%     end
end

diagnostic.residual  = residual;
diagnostic.S_rMSE    = s_rMSE;
diagnostic.objective = objective;

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

function [x,m,d] = Prox_Beckman(y,x0,rho,D_op,x,m,d,opts)
tau1    = opts.tau1;
tau2    = opts.tau2;
maxiter = opts.maxiter;
K_op = @(m,x) real(conj(D_op)*m) + x - x0;
K_op_mx = K_op(m,x);
for k = 1:maxiter
    prevm = m; prevx = x; prev_K_op_mx = K_op_mx;
    % Solve M (L2 norm shinkage)
    m = Prox_L21(prevm-tau1*conj(D_op')*d,tau1);
    % Solve x (Standard least-squares)
    x = pos( (rho*tau1)/(1+rho*tau1)*y + 1/(1+rho*tau1)*(prevx-tau1*d) ); % More proper
    % x = pos( (tau1*y + rho*prevx - tau1*rho*d)/(tau1+rho) );
    % solve d (over-relaxation)
    K_op_mx = K_op(m,x);
    d = d + tau2*( 2*K_op_mx - prev_K_op_mx );
end
end

function r = Prox_L1(r,rho)
r = sign(r).*max(0,abs(r)-rho);
end

function S = Prox_NonNeg_l1(A,rho)
S = max(0,A-rho);
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