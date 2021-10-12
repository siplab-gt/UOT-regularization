on_cluster = 0;
no_display = 0;

% Trial Parameters
nbr_trials = 20;
search_seeds = (1:10);
F1_threshold = 0.01;
eval_weights = [1,1,0];

% Target simulation
n = 10;
sim_param.imsize = [n,n];
sim_param.nbr_frames = 6;
sim_param.K = 0.05; % sparsity fraction
sim_param.B = 1; % Maximum distance of pixels between frames
sim_param.R = 0.10; % rank fraction of number of frames
sim_param.Q = 0.05; % percentage of targets to disappear
sim_param.magnitude = 'pos'; % targets all positive
sim_param.mass_growth_profile = 'tri';
sim_param.mass_growth_rate = 1.0;

% Measurement
sim_param.meas_method = 'cs';
sim_param.noise_sigma = 0.001; % zero-mean gaussian noise
sim_param.M = 0.60; % measurement fraction

% ADMM settings
ADMM_opts.sigma = 1.0;
ADMM_opts.rho = 0.5;
ADMM_opts.maxiter = 5000;
ADMM_opts.tolerance = 10^(-4);
ADMM_opts.beck_tau1 = 0.1;
ADMM_opts.beck_tau2 = 1.0;
ADMM_opts.beck_maxiter = 1;