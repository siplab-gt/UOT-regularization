% Figure011.m
% An experiment that solves the BPDN-UOT program of different sizes and
% records the timings and # iterations.

clearvars;
filename = 'Figure011';
ts_disp([filename ' started running.']);

%%


% Target simulation
% sim_param.imsize = [n,n];
sim_param.nbr_frames = 2;
sim_param.K = 0.08; % sparsity fraction
sim_param.B = 1; % Maximum distance of pixels between frames
sim_param.R = 0.10; % rank fraction of number of frames
sim_param.Q = 0.00; % percentage of targets to disappear
sim_param.magnitude = 'pos'; % targets all positive
sim_param.mass_growth_profile = 'static';
sim_param.mass_growth_rate = 0.0;

% Measurement
sim_param.M = 0.15;
sim_param.meas_method = 'iden';
sim_param.noise_sigma = 0.1; % zero-mean gaussian noise

% Define occlusion
sim_param.occl_size = [1.0,0.3]; % centered rectangle

% Define random seed
sim_param.rnd_seed = 1;

% Solvers to run
% dyn_prediction = 'previous_estimate';
dyn_prediction = 'ground_truth';

% Plotting options
markers = {'o','s','d','p','h','<','>'};

%% Perform simulation

% Experimental parameters
n_space = [8,16,32,64,128,256,512];
nbr_trials = 10;
walltime = nan(nbr_trials,length(n_space));
numiter = nan(nbr_trials,length(n_space));

% Algorithm parameters
lambda = 0.005;
kappa  = 0.002;
mu     = 2;
opts.rho = 0.1;
opts.maxiter = 2000;
opts.tolerance = 1e-3;
opts.beck_tau1 = 0.1;
opts.beck_tau2 = 1.0;
opts.beck_maxiter = 1;

%% Warm-up

n = min(n_space);
sim_param.imsize = [n,n];
t = 1;

% Generate simulation
[M_occ,X_gt,X_befocc,X_occ,Y,Phi] = run_simuation(n,t,sim_param);
compute_rMSE = @(x) norm(x-X_occ(:,2))^2 / norm(X_occ(:,2))^2;

% Perform solve
tic;
[x_hat,~,~,diagnostic] = solver_LS_UOT_Beckmann_ADMM([n,n],Y(:,2),Phi,X_gt(:,1),kappa,mu,opts);

%% Begin trials proper

for i = 1:length(n_space)
    % Set experimental variable
    n = n_space(i);
    sim_param.imsize = [n,n];
    
    % Perform a few solves and record statistics
    for t = 1:nbr_trials
        % Generate simulation
        [M_occ,X_gt,X_befocc,X_occ,Y,Phi] = run_simuation(n,t,sim_param);
        compute_rMSE = @(x) norm(x-X_occ(:,2))^2 / norm(X_occ(:,2))^2;

        % Perform solve
        tic;
        [x_hat,~,~,diagnostic] = solver_LS_UOT_Beckmann_ADMM([n,n],Y(:,2),Phi,X_gt(:,1),kappa,mu,opts);
        walltime(t,i) = toc;
        numiter(t,i) = length(find(~isnan(diagnostic.residual(:,1))));
        
        % Display
        disp(['n = ' num2str(n) ', '...
              't = ' num2str(t) ', '...
              'rMSE = ' num2str(compute_rMSE(x_hat)) ', '...
              'time = ' num2str(walltime(t,i)) ', '...
              'niter = ' num2str(numiter(t,i)) ', '...
              ]);
    end
end

save(filename,'n_space','walltime','numiter');

%% Plotting

fig = figure(1); set(fig,'Units','normalized','Position',[0.1 0.1 0.25 0.3]); clf;
ha = tight_subplot(3,1,[.03 .01],[.15 .03],[.10 .03]);

axes(ha(1)); 
errorbar(n_space.^2,median(walltime),median(walltime)-prctile(walltime,25),prctile(walltime,75)-median(walltime),'-o','LineWidth',2);
set(gca, 'XScale', 'log', 'YScale', 'log');
ylabel('Wall time (s)','Interpreter','LaTex');
grid on; axis tight; set(gca,'XTickLabel',[]);
yticks(logspace(-2,2,5));

axes(ha(2)); errorbar(n_space.^2,median(numiter),median(numiter)-prctile(numiter,25),prctile(numiter,75)-median(numiter),'-o','LineWidth',2);
set(gca, 'XScale', 'log', 'YScale', 'log');
yticks(100:100:500);
ylabel('Iterations','Interpreter','LaTex');
grid on; axis tight; set(gca,'XTickLabel',[]);

axes(ha(3)); 
plot(n_space.^2,median(walltime)./median(numiter),'-o','LineWidth',2); grid on;
set(gca, 'XScale', 'log', 'YScale', 'log');
xlabel('Number of pixels','Interpreter','LaTex');
ylabel('Per iteration time (s)','Interpreter','LaTex');
grid on; axis tight;

drawnow; saveFig2PDF(filename);

%% Simulation and solver settings

function [M_occ,X_gt,X_befocc,X_occ,Y,Phi] = run_simuation(n,rnd_seed,sim_param)
% Create an occlusion mask
M_occ = zeros(sim_param.imsize);
M_occ(max(1,floor( sim_param.imsize(1)*(0.5-sim_param.occl_size(1)/2) )) : ...
      min(n,ceil( sim_param.imsize(1)*(0.5+sim_param.occl_size(1)/2) )) , ...
      max(1,floor( sim_param.imsize(2)*(0.5-sim_param.occl_size(2)/2) )) : ...
      min(n,ceil( sim_param.imsize(2)*(0.5+sim_param.occl_size(2)/2) )) ) = -2;

rng(rnd_seed,'Twister');

X_gt = simulate_pixels([n,n], sim_param.nbr_frames, ceil(sim_param.K*n*n), sim_param.B, sim_param.magnitude, sim_param.mass_growth_profile, sim_param.mass_growth_rate);

% Apply occlusion
X_befocc = X_gt + M_occ(:)*ones(1,sim_param.nbr_frames);
X_occ = max(X_befocc,0);

switch sim_param.meas_method
case 'iden'
    Phi = speye(n^2);
    Y = X_occ + sim_param.noise_sigma*randn(size(X_occ));
case 'cs'
    M = ceil(sim_param.M*sim_param.imsize(1)*sim_param.imsize(2));
    [Y, Phi] = take_gaussian_meas(X_occ, M, sim_param.noise_sigma^2);
case 'gblur'
    m = 3;
    Phi0 = convmtx2( fspecial('gaussian',[m,m]) ,sim_param.imsize(1),sim_param.imsize(2));
    Mask = ones(sim_param.imsize(1)+m-1,sim_param.imsize(2)+m-1);
    Mask(floor(m/2)+1:end-ceil(m/2)+1,floor(m/2)+1:end-ceil(m/2)+1) = zeros(sim_param.imsize);
    Phi0(find(Mask),:) = [];
    Phi = zeros(sim_param.imsize(1)*sim_param.imsize(2),sim_param.imsize(1)*sim_param.imsize(2),sim_param.nbr_frames);
    for f = 1:sim_param.nbr_frames
        Phi(:,:,f) = Phi0; 
        Y(:,f) = Phi0*X_occ(:,f) + sim_param.noise_sigma*randn(size(X_occ(:,f)));
    end
end
end

