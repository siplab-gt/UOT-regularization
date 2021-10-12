% Figure007_v2.m
% Simple experiment to highlight pros and cons of different OT algorithms.

clearvars;
filename = 'Figure007_v2';
ts_disp([filename ' started running.']);

%% 

% Target simulation
n = 10;
sim_param.imsize = [n,n];
sim_param.nbr_frames = 2;
sim_param.K = 0.05; % sparsity fraction
sim_param.B = 1; % Maximum distance of pixels between frames
sim_param.R = 0.10; % rank fraction of number of frames
sim_param.Q = 0.05; % percentage of targets to disappear
sim_param.magnitude = 'pos'; % targets all positive
sim_param.mass_growth_profile = 'static';
sim_param.mass_growth_rate = 1.0;

% Measurement
% sim_param.M = [];
% sim_param.meas_method = 'iden';
sim_param.M = 0.35; % sparsity fraction
sim_param.meas_method = 'cs';
sim_param.noise_sigma = 0.01; % zero-mean gaussian noise

% Solvers to run
solvers = {'Balanced-OT','Unbalanced-OT'};

% Scenario
scenarios = {'Growth','Decay'};

%% Experimental variables

% Figure: _qualitative
variable_name = 'Noise level (\sigma)';
variable_space = 0.01;
sim_param.mass_growth_rate = 1.5;
figs_disp = [1,0];

% % Figure: _varyingnoise
% variable_name = 'Noise level (\sigma)';
% variable_space = logspace(-3,0,10);
% figs_disp = [0,1];

% % Figure: _varyingrate
% variable_name = 'Rate of mass change';
% variable_space = 0.0:0.2:1.8;
% figs_disp = [0,1];

%% Run Trials

% Initialize storage
global_rMSE = nan(length(scenarios),length(solvers),length(variable_space));
global_F1 = nan(length(scenarios),length(solvers),length(variable_space));

% Display options
fontsize = 14;
legend_frame = 1;
markers = {'o','s','d','^'};
clims = [0,(1 + sim_param.mass_growth_rate/2)];

% Setup figures
if figs_disp(1)
fig1 = figure(1); clf; 
set(fig1,'Units','normalized','Position',[0.1 0.1 0.35 0.35]); clf;
ga = tight_subplot(length(scenarios),2+length(solvers),[.08 .01],[.08 .08],[.04 .01]);
end

for sc = 1:length(scenarios)
    
    for v = 1:length(variable_space)
        
	% Set experimental variable
    switch variable_name
    case 'Noise level (\sigma)'
        sim_param.noise_sigma = variable_space(v);
        subfilename = 'varyingnoise';
    case 'Rate of mass change'
        sim_param.mass_growth_rate = variable_space(v);
        subfilename = 'varyingrate';
    end

    % Simulate Problem
    rng(1,'Twister'); [~,Phi,X_gt] = simulator(sim_param);
    switch scenarios{sc}
    case 'Growth', X_gt(:,1) = X_gt(:,1) * (1 - sim_param.mass_growth_rate/2);
    case 'Decay',  X_gt(:,1) = X_gt(:,1) * (1 + sim_param.mass_growth_rate/2);
    end
    for t = 1:size(Phi,3)
        Y_gt(:,t) = Phi(:,:,t) * X_gt(:,t) + sim_param.noise_sigma * randn(size(Phi(:,:,t),1),1);
    end
    y = Y_gt(:,2);
    Phi = Phi(:,:,2);
    x0 = X_gt(:,1);
    compute_rMSE = @(x) norm(vec(x)-X_gt(:,2))^2 / norm(vec(X_gt(:,2)))^2;

    if figs_disp(1)
    figure(fig1);
    axes(ga((sc-1)*(2+length(solvers))+1)); cla;
    imagesc(reshape(X_gt(:,1),sim_param.imsize),clims); axis image;
    if sc == 1, title('Prior $s_0$','FontSize',fontsize,'Interpreter','LaTex'); end
    set(gca,'xticklabel',[],'yticklabel',[],'xtick',[],'ytick',[]);
    xlabel(['(Magnitude = ' num2str(max(X_gt(:,1))) ')'],'FontSize',fontsize,'Interpreter','LaTex');
    axes(ga((sc-1)*(2+length(solvers))+2)); cla;
    imagesc(reshape(X_gt(:,2),sim_param.imsize),clims); axis image; 
    if sc == 1, title('Ground Truth $s^\star$','FontSize',fontsize,'Interpreter','LaTex'); end
    set(gca,'xticklabel',[],'yticklabel',[],'xtick',[],'ytick',[]);
    xlabel(['(Magnitude = ' num2str(max(X_gt(:,2))) ')'],'FontSize',fontsize,'Interpreter','LaTex');
    end
    
    % Run Solvers
    for s = 1:length(solvers)
%         ts_disp(['Running trials for ' solvers{s}]);
        switch solvers{s}
        case 'Balanced-OT'
            % Define search space
            kappa_space = logspace(-2,1,13);
            mu_space = 0;
            [kappa_grid,mu_grid] = meshgrid(kappa_space,0);
            run_solver = @(kappa,mu) solver_Balanced(sim_param.imsize,y,Phi,x0,kappa);
        case 'Unbalanced-OT'
            % Define search space
            kappa_space = logspace(-2,1,13);
            mu_space = logspace(-2,+2,17);
            [kappa_grid,mu_grid] = meshgrid(kappa_space,mu_space);
            run_solver = @(kappa,mu) solver_Unbalanced(sim_param.imsize,y,Phi,x0,kappa,mu);
        end

        % Perform parallel solves
        X_parfor = zeros(sim_param.imsize(1)*sim_param.imsize(2), numel(kappa_grid));
        parfor i = 1:numel(kappa_grid)
            X_parfor(:,i) = run_solver(kappa_grid(i),mu_grid(i));
        end
        % Reorganize to save
        rMSE = nan( size(kappa_grid) ); F1 = nan( size(kappa_grid) );
        for i = 1:numel(kappa_grid)
            results(sc,s,i).kappa   = kappa_grid(i);
            results(sc,s,i).mu      = mu_grid(i);
            results(sc,s,i).solver  = solvers{s};
            results(sc,s,i).scenario = scenarios{sc};
            results(sc,s,i).sigma   = sim_param.noise_sigma;
            results(sc,s,i).x       = X_parfor(:,i);
            rMSE(i)     = compute_rMSE( X_parfor(:,i) );
            F1(i)       = compute_F1( X_parfor(:,i) , X_gt(:,2) , 0.1 );
        end

        % Extract best performing
        [~,rMSEmin_idx] = min( rMSE(:) );
        [m,k] = ind2sub( [length(mu_space),length(kappa_space)] , rMSEmin_idx );
        x_best = X_parfor(:,rMSEmin_idx);
        global_rMSE(sc,s,v) = rMSE(rMSEmin_idx);
        [~,F1min_idx] = max( F1(:) );
        global_F1(sc,s,v)   = F1(F1min_idx);
        
        % Display
        if figs_disp(1)
        figure(fig1);
        axes(ga((sc-1)*(2+length(solvers))+2+s));
        imagesc(reshape(x_best,sim_param.imsize),clims); axis image;
        colormap(flipud(gray));
        set(gca,'xticklabel',[],'yticklabel',[],'xtick',[],'ytick',[]);
        if sc == 1, title(solvers{s},'FontSize',fontsize,'Interpreter','LaTex'); end
        xlabel(['(rMSE = ' num2str(rMSE(rMSEmin_idx),'%04.3f') ')'],'FontSize',fontsize,'Interpreter','LaTex');
        end
        
        drawnow;
        
        ts_disp([variable_name '=' num2str(variable_space(v)) ', '...
                  scenarios{sc} ', ' solvers{s} ', '...
                 'rMSE=' num2str(global_rMSE(sc,s,v)) ', '...
                 'F1=' num2str(global_F1(sc,s,v)) ]);
    end
    end
end

if figs_disp(1)
figure(fig1);
% colormap(flipud(gray));
for sc = 1:length(scenarios) 
    axes(ga((sc-1)*(2+length(solvers))+1)); ylabel(scenarios{sc}); 
    ylabel(scenarios{sc},'FontSize',fontsize,'Interpreter','LaTex');
end
drawnow; saveFig2PDF([filename '_qualitative']);
end

%% rMSE plots

if figs_disp(2)
fig2 = figure(2); clf; 
set(fig2,'Units','normalized','Position',[0.1 0.1 0.4 0.3]); clf;
ja = tight_subplot(1,length(scenarios),[.01 .06],[.15 .1],[.08 .03]);

axes(ja(1)); cla;
switch variable_name
case 'Noise level (\sigma)'
    for s = 1:length(solvers), loglog(variable_space,squeeze(global_rMSE(1,s,:)),'-','LineWidth',3,'Marker',markers{s}); hold on; end
    title(['Mass ' scenarios{1}],'FontSize',fontsize);
    legend(solvers,'Location','SouthEast','FontSize',fontsize);
case 'Rate of mass change'
    for s = 1:length(solvers), semilogy(variable_space,squeeze(global_rMSE(1,s,:)),'-','LineWidth',3,'Marker',markers{s}); hold on; end
end
xlim([variable_space(1), variable_space(end)]);
ylabel('rMSE','FontSize',fontsize);
xlabel(variable_name,'FontSize',fontsize);
axis tight; grid on;

axes(ja(2)); cla;
switch variable_name
case 'Noise level (\sigma)'
    for s = 1:length(solvers), loglog(variable_space,squeeze(global_rMSE(2,s,:)),'-','LineWidth',3,'Marker',markers{s}); hold on; end
    title(['Mass ' scenarios{2}],'FontSize',fontsize);
case 'Rate of mass change'
    for s = 1:length(solvers), semilogy(variable_space,squeeze(global_rMSE(2,s,:)),'-','LineWidth',3,'Marker',markers{s}); hold on; end
end
xlim([variable_space(1), variable_space(end)]);
xlabel(variable_name,'FontSize',fontsize);
set(gca,'yticklabel',[]);
axis tight; grid on;

linkaxes(ja,'y'); % Link all subplots xlim
drawnow; saveFig2PDF([filename '_' subfilename]);
end

%% Solvers

function D = GenerateDivergenceMatrices(imsize)
Dx = eye(imsize(1)*imsize(2)) - circshift(eye(imsize(1)*imsize(2)),1);
Dx(1,imsize(1)*imsize(2)) = 0;
Dx(:,imsize(1):imsize(1):imsize(1)*imsize(2)) = zeros(imsize(1)*imsize(2),imsize(2));
Dy = eye(imsize(1)*imsize(2)) - circshift(eye(imsize(1)*imsize(2)),imsize(1));
Dy(:,end-imsize(1)+1:end) = zeros(imsize(1)*imsize(2),imsize(1));
D = Dx + 1i*Dy;
end

function x = solver_Balanced(imsize,y,Phi,x0,kappa)
opts.rho = 0.1;
opts.maxiter = 1000;
opts.tolerance = 1e-5;
opts.beck_tau1 = 0.1;
opts.beck_tau2 = 1.0;
opts.beck_maxiter = 1;
x = solver_LS_BOT_Beckmann_ADMM(imsize,y,Phi,x0,kappa,opts);
end

function x = solver_Unbalanced(imsize,y,Phi,x0,kappa,mu)
opts.rho = 0.1;
opts.maxiter = 1000;
opts.tolerance = 1e-5;
opts.beck_tau1 = 0.1;
opts.beck_tau2 = 1.0;
opts.beck_maxiter = 1;
x = solver_LS_UOT_Beckmann_ADMM(imsize,y,Phi,x0,kappa,mu,opts);
end