% Figure005.m
% Algorithm performance vs size of batch

clearvars;
filename = 'Figure005';
ts_disp([filename ' started running.']);

%% Parameters

% Search parameters
% redo_search = [0,0,0,0]; load([filename '.mat']); % comment if full search required

% Default parameters
set_default_parameters;

% Trial Parameters
set_trial_parameters;

% Solvers to run
solvers = {'RPCA','RPCA+L1-DF','RPCA+BOT-DF','RPCA+UOT-DF'};

%% Run trials

if on_cluster, PaceParalleltoolbox_r2016b(true); end

for i = 1:length(nbr_frames_space)
    % Update trial variable
    sim_param.nbr_frames = nbr_frames_space(i);
    
    % Parameter search (efficiently using parallelization)
    if ~exist('redo_search')
        ts_disp('Running full parameter search.');
        solver_param = get_solver_param_via_search(solvers,search_seeds,F1_threshold,eval_weights,sim_param,ADMM_opts);
    else
        for s = 1:length(redo_search)
        if redo_search(s)
            ts_disp(['Running parameter search for ' solvers{s} '.']);
            solver_param(s) = get_solver_param_via_search(cellstr(solvers{s}),search_seeds,F1_threshold,eval_weights,sim_param,ADMM_opts);
        else
            ts_disp(['Retrieving parameters from ' solvers{s} ' : '...
                     'lambda = ' num2str(results(s,i,1).param.lambda) ', '...
                     'gamma = ' num2str(results(s,i,1).param.gamma) ', '...
                     'kappa = ' num2str(results(s,i,1).param.kappa) ', '...
                     'mu = ' num2str(results(s,i,1).param.mu) ...
                     ]);
            solver_param(s) = results(s,i,1).param;
        end
        end
    end

    % Discretize the solvers and trials space (for parallelization)
    [t_grid,s_grid] = meshgrid( 1:nbr_trials , 1:length(solvers) );

    % Run trials (efficiently using parallelization)
    ts_disp(['Running parallel trials...']);
    S_parfor = zeros(sim_param.imsize(1)*sim_param.imsize(2), sim_param.nbr_frames, numel(s_grid));
    L_parfor = zeros(sim_param.imsize(1)*sim_param.imsize(2), sim_param.nbr_frames, numel(s_grid));
    time_parfor = zeros(numel(s_grid),1);
    parfor j = 1:numel(s_grid)
        s = s_grid(j); t = t_grid(j);
        % Simulate Problem
        rng(t,'twister'); [Y,Phi] = simulator(sim_param);
        % Run solver
        [S_parfor(:,:,j),L_parfor(:,:,j),time_parfor(j)] = ...
            run_solver(solvers{s},sim_param.imsize,Y,Phi,solver_param(s),ADMM_opts);
    end
    
    % Save metrics
    time   = zeros(length(solvers),nbr_trials);
    F1     = zeros(length(solvers),nbr_trials);
    S_rMSE = zeros(length(solvers),nbr_trials);
    L_rMSE = zeros(length(solvers),nbr_trials);
    for j = 1:numel(s_grid)
        s = s_grid(j); t = t_grid(j);
        
        % Simulate Problem
        rng(t,'twister'); [~,~,S_gt,L_gt] = simulator(sim_param);
        
        % Save data
        results(s,i,t).S = S_parfor(:,:,j);
        results(s,i,t).L = L_parfor(:,:,j);
        results(s,i,t).time = time_parfor(j);
        results(s,i,t).solver = solvers{s};
        results(s,i,t).param = solver_param(s);

        % Evaluation metrics
        time(s,t)   = time_parfor(j);
        F1(s,t)     = compute_F1(results(s,i,t).S,S_gt,F1_threshold);
        S_rMSE(s,t) = norm(vec(results(s,i,t).S-S_gt))^2 / norm(vec(S_gt))^2;
        L_rMSE(s,t) = norm(vec(results(s,i,t).L-L_gt))^2 / norm(vec(L_gt))^2;

%         % Display in terminal
%         disp([var_name ' = ' num2str(sim_param.noise_sigma) ' ' ...
%               'Trial #' num2str(t) '/' num2str(nbr_trials) '- ' ...
%               solvers{s} ' (time=' num2str(results(s,i,t).time) 's) : '...
%               'F1 = ' num2str( F1(s,t) ) ', '...
%               'S_rMSE = ' num2str( S_rMSE(s,t) ) ', '...
%               'L_rMSE = ' num2str( L_rMSE(s,t) ) ]);   
    end
    
    % Display summmary statistics
    ts_disp(['Summary statistics of ' num2str(size(results,3)) ' trials '...
             'for ' var_name ' = ' num2str(sim_param.nbr_frames)]);
    fprintf('Method\t\t\tTime\t\tF1 score\tS_rMSE\t\tL_rMSE\n');
    for s = 1:length(solvers)
        switch solvers{s}
            case 'RPCA', fprintf([solvers{s} '\t\t\t']);
            otherwise, fprintf([solvers{s} '\t\t']);
        end
        fprintf('%08.4f\t',median(time(s,:)));
        fprintf('%08.4f\t',mean(F1(s,:)));
        fprintf('%08.4f\t',mean(S_rMSE(s,:)));
        fprintf('%08.4f\t\n',mean(L_rMSE(s,:)));
    end
    
    % Display Progress
    ts_disp(['Progress of ' filename ' : ' num2str(i/length(nbr_frames_space)*100) '% completed.']);
end

% Save Data
clear S_parfor L_parfor time_parfor S_gt L_gt F1 S_rMSE L_rMSE time
close all; save(filename);

if no_display, return; end

%% Generate figure

% Compute metrics
S_rMSE  = nan(length(solvers),length(nbr_frames_space),nbr_trials);
L_rMSE  = nan(length(solvers),length(nbr_frames_space),nbr_trials);
F1      = nan(length(solvers),length(nbr_frames_space),nbr_trials);
runtime = nan(length(solvers),length(nbr_frames_space),nbr_trials);
for i = 1:length(nbr_frames_space)
    sim_param.nbr_frames = nbr_frames_space(i); % Update trial variable
    for t = 1:nbr_trials
    for s = 1:length(solvers)
        % Simulate Problem
        rng(t,'twister'); [~,~,S_gt,L_gt] = simulator(sim_param);
        
        % Evaluation metrics
        compute_S_rMSE = @(S) norm(vec(S-S_gt))^2 / norm(vec(S_gt))^2;
        compute_L_rMSE = @(L) norm(vec(L-L_gt))^2 / norm(vec(L_gt))^2;
        compute_F1_score = @(S) compute_F1(S,S_gt,F1_threshold);

        % Compute
        S_rMSE(s,i,t)   = compute_S_rMSE(results(s,i,t).S);
        L_rMSE(s,i,t)   = compute_L_rMSE(results(s,i,t).L);
        F1(s,i,t)       = compute_F1_score(results(s,i,t).S);
        runtime(s,i,t)  = results(s,i,t).time;
    end
    end
end

% Plot options
fontSize = 14;
markers = {'o','s','d','^'};
cmap = [     0    0.4470    0.7410;
        0.8500    0.3250    0.0980;
        0.9290    0.6940    0.1250;
        0.4940    0.1840    0.5560;
        0.4660    0.6740    0.1880;
        0.3010    0.7450    0.9330;
        0.6350    0.0780    0.1840];

% Plot
fig = figure(1); clf;
set(fig,'Units','normalized','Position',[0.1 0.0 0.3 1.0]); clf;

S_rMSE_lines = []; L_rMSE_lines = []; F1_lines = []; runtime_lines = [];
for s = 1:length(solvers)
    subplot(411);
%     h(1,s).mainLine = errorbar(nbr_frames_space,mean(squeeze(rMSE(s,:,:))'),std(squeeze(rMSE(s,:,:))'),...
%                                'LineWidth',2,'Color',cmap(s,:),'Marker',markers{s});
    h(1,s) = shadedErrorBar(nbr_frames_space,squeeze(S_rMSE(s,:,:))',{@nanmean,@nanstd},...
                            'lineprops',{['-' markers{s}],'LineWidth',2,'Color',cmap(s,:),'MarkerFaceColor',cmap(s,:)}); hold on;
    S_rMSE_lines = [S_rMSE_lines, h(1,s).mainLine];
    hold on;
    
    subplot(412);
%     h(1,s).mainLine = errorbar(nbr_frames_space,mean(squeeze(rMSE(s,:,:))'),std(squeeze(rMSE(s,:,:))'),...
%                                'LineWidth',2,'Color',cmap(s,:),'Marker',markers{s});
    h(2,s) = shadedErrorBar(nbr_frames_space,squeeze(L_rMSE(s,:,:))',{@nanmean,@nanstd},...
                            'lineprops',{['-' markers{s}],'LineWidth',2,'Color',cmap(s,:),'MarkerFaceColor',cmap(s,:)}); hold on;
    L_rMSE_lines = [L_rMSE_lines, h(2,s).mainLine];
    hold on;
    
    subplot(413);
%     h(2,s).mainLine = errorbar(nbr_frames_space,mean(squeeze(F1(s,:,:))'),std(squeeze(F1(s,:,:))'),...
%                                'LineWidth',2,'Color',cmap(s,:),'Marker',markers{s});
    h(3,s) = shadedErrorBar(nbr_frames_space,squeeze(F1(s,:,:))',{@nanmean,@nanstd},...
                            'lineprops',{['-' markers{s}],'LineWidth',2,'Color',cmap(s,:),'MarkerFaceColor',cmap(s,:)}); hold on;
    F1_lines = [F1_lines, h(3,s).mainLine];
    hold on;
    
    subplot(414);
%     h(2,s).mainLine = errorbar(nbr_frames_space,mean(squeeze(runtime(s,:,:))'),std(squeeze(F1(s,:,:))'),...
%                                'LineWidth',2,'Color',cmap(s,:),'Marker',markers{s});
    h(3,s) = shadedErrorBar(nbr_frames_space,squeeze(runtime(s,:,:))',{@nanmean,@nanstd},...
                            'lineprops',{['-' markers{s}],'LineWidth',2,'Color',cmap(s,:),'MarkerFaceColor',cmap(s,:)}); hold on;
    runtime_lines = [runtime_lines, h(3,s).mainLine];
    hold on;
    
end

subplot(411);
% loglog(nbr_frames_space,mean(rMSE,3)','LineWidth',2);
axis tight; grid on; grid minor; ylim([0,1]);
% set(gca, 'XScale', 'log', 'YScale', 'log'); ylim([0,1]);
% set(gca, 'XScale', 'log'); 
% xlabelvar_name,'FontSize',fontSize);
ylabel('Signal error','FontSize',fontSize);
% legend(S_rMSE_lines,solvers,'FontSize',fontSize,'Location','SouthWest');

subplot(412);
% loglog(nbr_frames_space,mean(rMSE,3)','LineWidth',2);
axis tight; grid on; grid minor; ylim([0,0.1]);
% set(gca, 'XScale', 'log', 'YScale', 'log'); ylim([0,1]);
% set(gca, 'XScale', 'log');
% xlabel(var_name,'FontSize',fontSize);
ylabel('Low rank error','FontSize',fontSize);
% legend(L_rMSE_lines,solvers,'FontSize',fontSize,'Location','NorthWest');

subplot(413); 
%loglog(nbr_frames_space,mean(F1,3)','LineWidth',2);
axis tight; grid on; grid minor; ylim([0,1]);
% set(gca, 'XScale', 'log'); 
% xlabel(var_name,'FontSize',fontSize);
ylabel('Support estimation','FontSize',fontSize);
% legend(F1_lines,solvers,'FontSize',fontSize,'Location','SouthWest');

subplot(414); 
%loglog(nbr_frames_space,mean(runtime,3)','LineWidth',2);
axis tight; grid on; grid minor; %ylim([0,1]);
% set(gca, 'XScale', 'log'); 
xlabel(var_name,'FontSize',fontSize);
ylabel('Run time (s)','FontSize',fontSize);
% legend(F1_lines,solvers,'FontSize',fontSize,'Location','SouthWest');

drawnow; saveFig2PDF(filename);