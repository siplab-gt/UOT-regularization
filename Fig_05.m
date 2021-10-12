% Figure008.m
% This experiment 

clearvars;
this_filename = 'Figure008';
ts_disp([this_filename ' started running.']);

%% Parameters

% Default parameters
set_default_parameters;

% Trial settings
beck_maxiter_space = 1:30;
var_name = 'Number of Beckmann Prox Iterations';

% ADMM settings
this_ADMM_opts.sigma = 1.0;
this_ADMM_opts.rho = 0.5;
this_ADMM_opts.maxiter = 5000;
this_ADMM_opts.tolerance = 10^(-4);
this_ADMM_opts.beck_tau1 = 0.1;
this_ADMM_opts.beck_tau2 = 1.0;
this_ADMM_opts.beck_maxiter = 1;

%% Simulate Problem

search_seeds = 1;
rng(search_seeds,'Twister'); [Y,Phi,this_ADMM_opts.S_gt,this_ADMM_opts.L_gt] = simulator(sim_param);

% Run solver
load('Figure000.mat');
idx = find(strcmp(solvers,'RPCA+UOT-DF'));
lambda = solver_param(idx).lambda;
gamma  = solver_param(idx).gamma;
kappa  = solver_param(idx).kappa;
mu     = solver_param(idx).mu;

% Run Trials
for i = 1:length(beck_maxiter_space)
    % Update trial parameter
    this_ADMM_opts.beck_maxiter = beck_maxiter_space(i);
    
    % Run solver
    [~,~,~,~,diagnostic(i)] = ...
        solver_RPCA_UOT_Beckman_ADMM(sim_param.imsize,Y,Phi,lambda,gamma,kappa,mu,this_ADMM_opts);
    
    % Display
    numiter = length(find(~isnan(diagnostic(i).objective)));
    ts_disp(['Beck maxiter = ' num2str(this_ADMM_opts.beck_maxiter) ...
             ' completed in ' num2str(numiter) ' ADMM iterations, '...
             'rMSE = ' num2str(diagnostic(i).S_rMSE(numiter)) ', '...
             'obj = ' num2str(diagnostic(i).objective(numiter)) ]);
end

%% Parse data for plotting

tolerance_space = [1e-4];

for t = 1:length(tolerance_space)
for i = 1:length(beck_maxiter_space)
    % Find termination point
    tolerance = tolerance_space(t);

    idx = min(find(diagnostic(i).residual(:,1)<tolerance & ...
                   diagnostic(i).residual(:,2)<tolerance));
    if isempty(idx), idx = size(diagnostic(i).residual(:,1),1); end
    
    Iters(t,i)  = idx;
    S_rMSE(t,i) = diagnostic(i).S_rMSE(idx);
    L_rMSE(t,i) = diagnostic(i).L_rMSE(idx);
    objective(t,i) = diagnostic(i).objective(idx);
end
end

fig = figure; set(fig,'Units','normalized','Position',[0.1 0.1 0.3 0.25]); clf;
ha = tight_subplot(1,1,[.03 .01],[.15 .03],[.10 .03]);

axes(ha(1));
plot((Iters'/min(Iters(:))-1)*100,'-o','LineWidth',3);
% ylabel('\% $\Delta$ of \# ADMM Iterations','Interpreter','LaTex');
ylabel('Additional $\%$ of ADMM iterations','Interpreter','LaTex');
xlabel('\# Beckmann Proximal Iterations','Interpreter','LaTex');
grid on; axis tight;

drawnow; saveFig2PDF(this_filename);
