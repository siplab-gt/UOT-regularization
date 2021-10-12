% Figure005.m
% Qualitative example between algorithms
% Algorithms: (1) RPCA (2) RPCA+L1-DF (3) RPCA+UBOT-DF

clearvars;
filename = 'Figure000';
ts_disp([filename ' started running.']);

%% Parameters

% Default parameters
set_default_parameters;

% Solvers to run
solvers = {'RPCA','RPCA+L1-DF','RPCA+BOT-DF','RPCA+UOT-DF'};

%% Simulate Problem

search_seeds = 1;
rng(search_seeds,'Twister'); [Y,Phi,S_gt,L_gt] = simulator(sim_param);
compute_S_rMSE = @(S) norm(vec(S-S_gt))^2 / norm(vec(S_gt))^2;
compute_L_rMSE = @(L) norm(vec(L-L_gt))^2 / norm(vec(L_gt))^2;

%% Obtain parameters

if on_cluster, PaceParalleltoolbox_r2016b(true); end

% Run full search (takes 10 mins)
solver_param = get_solver_param_via_search(solvers,search_seeds,F1_threshold,eval_weights,sim_param,ADMM_opts);

%% Run solvers

for s = 1:length(solvers)
    % Run solver
    [results(s).S,results(s).L,solve_time] = ...
        run_solver(solvers{s},sim_param.imsize,Y,Phi,solver_param(s),ADMM_opts);
    
    % Save data
    results(s).solver = solvers{s};
    results(s).param = solver_param(s);
    results(s).F1 = compute_F1(results(s).S,S_gt,F1_threshold);
    results(s).S_rMSE = compute_S_rMSE(results(s).S);
    results(s).L_rMSE = compute_L_rMSE(results(s).L);
    
    % Display on terminal
    disp([solvers{s} ' (time=' num2str(solve_time) 's) : '...
          'F1 = ' num2str( results(s).F1 ) ', '...
          'S_rMSE = ' num2str(results(s).S_rMSE) ', '...
          'L_rMSE = ' num2str(results(s).L_rMSE) ]);
end

% Save Data
close all; save(filename);

if on_cluster, return; end

%% Generate figure

% Plot options
fontSize = 10;
borderline_color = 'k';

fig = figure(1);
set(fig,'Units','normalized','Position',[0.1 0.1 0.3 0.5]); clf;
ha = tight_subplot(length(solvers)+1,1,[.01 .01],[.01 .01],[.01 .01]);

axes(ha(1));
display_batch(S_gt,sim_param.imsize); 
axis on; ylabel('Ground Truth','FontSize',fontSize);
set(gca,'xticklabel',[],'yticklabel',[],'xtick',[],'ytick',[]);

for s = 1:length(solvers)
    axes(ha(s+1));
    display_batch(results(s).S,sim_param.imsize,[0,max(S_gt(:))]);
    axis on; ylabel(solvers{s},'FontSize',fontSize);
    set(gca,'xticklabel',[],'yticklabel',[],'xtick',[],'ytick',[]);
end

colormap(flipud(gray));
drawnow; saveFig2PDF(filename);