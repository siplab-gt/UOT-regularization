% Figure010.m
% Simple experiment to illustrate benefits of OT algorithms under an 
% occlusion settings.

clearvars;
filename = 'Figure010';
ts_disp([filename ' started running.']);

%% Freeze the results in paper

% Target simulation
n = 16;
sim_param.imsize = [n,n];
sim_param.nbr_frames = 31;
sim_param.K = 0.08; % sparsity fraction
sim_param.B = 1; % Maximum distance of pixels between frames
sim_param.R = 0.10; % rank fraction of number of frames
sim_param.Q = 0.00; % percentage of targets to disappear
sim_param.magnitude = 'pos'; % targets all positive
sim_param.mass_growth_profile = 'static';
sim_param.mass_growth_rate = 0.0;

% Measurement
sim_param.M = 0.12;
sim_param.meas_method = 'cs';
sim_param.noise_sigma = 0.05; % zero-mean gaussian noise

% Define occlusion
sim_param.occl_size = [1.0,0.3]; % centered rectangle

% Define random seed
sim_param.rnd_seed = 3;

% Solvers to run
% L1        : LS + sparsity + dynamics
% RWL1      : L1 + sparsity dynamics
% BOT       : LS + sparsity + BOT dynamics
% UOT       : LS + sparsity + UOT dynamics
solvers = {'L1','RWL1','BOT','UOT'}; % for presentation

% dyn_prediction = 'ground_truth';
dyn_prediction = 'previous_estimate';

% Plotting options
markers = {'o','s','d','p','h','<','>'};

%% Perform simulation

[M_occ,X_gt,X_befocc,X_occ,Y,Phi] = run_simuation(n,sim_param.rnd_seed,sim_param);

%% Exhaustive algorithmic parameter search

fig1 = figure(100); clf;
set(fig1,'Units','normalized','Position',[0.1 0.1 0.4 0.4]); clf;
ha = tight_subplot(length(solvers)+2,sim_param.nbr_frames-1,[.01 .01],[.01 .01],[.01 .01]);
for f = 1:sim_param.nbr_frames-1
    axes(ha(f)); imagesc(reshape(X_befocc(:,f+1),[n,n])); axis equal tight off;
    axes(ha(f+(sim_param.nbr_frames-1))); imagesc(reshape(X_occ(:,f+1),[n,n])); axis equal tight off;
end
fig2 = figure(101); clf;
set(fig2,'Units','normalized','Position',[0.5 0.5 0.4 0.25]);
grid on; hold on;
drawnow;
fig3 = figure(102); clf;
set(fig3,'Units','normalized','Position',[0.5 0.1 0.4 0.25]);
grid on; hold on;
cord = get(gca,'colororder');
drawnow;

for s = 1:length(solvers)
    % Configure settings for each solver
    [lambda_space,kappa_space,mu_space,solve_frame] = configure_solver(solvers{s},sim_param);
    run_solver = @(lambda,kappa,mu) solve_video(solve_frame,Y,Phi,X_occ,lambda,kappa,mu,dyn_prediction);
    compute_rMSE = @(X_hat) norm(X_hat-X_occ(:,2:end),'fro')^2 / norm(X_occ(:,2:end),'fro')^2; % overall min
    % compute_rMSE = @(X_hat) median(sum((X_hat-X_occ(:,2:end)).^2)./sum((X_occ(:,2:end)).^2)); % median;
    
    % Run exhaustive search in parallel
    [lambda_grid,kappa_grid,mu_grid] = meshgrid(lambda_space,kappa_space,mu_space);
    X_parfor = zeros(size(X_gt,1),size(X_gt,2)-1, numel(kappa_grid));
    rMSE_parfor = nan(size(kappa_grid));
    for i = 1:numel(kappa_grid)
        % Compute
        X_parfor(:,:,i) = run_solver(lambda_grid(i),kappa_grid(i),mu_grid(i));
        rMSE_parfor(i) = compute_rMSE(X_parfor(:,:,i));
        
        % Plot
        figure(s); clf;
        subplot(221); videosc([n,n],X_parfor(:,:,i));
        title(['\lambda = ' num2str(lambda_grid(i)) ', '...
               '\kappa = ' num2str(kappa_grid(i)) ', '...
               '\mu = ' num2str(mu_grid(i)) ', '...
               'rMSE = ' num2str(rMSE_parfor(i))]);
        subplot(222); videosc([n,n],X_occ(:,2:end));
        % subplot(223); imagesc(log10(rMSE_parfor)); colorbar;
        subplot(223); imagesc(log10(rMSE_parfor(:,:,find(mu_space==mu_grid(i))))); colorbar;
        xlabel('\lambda'); ylabel('\kappa');
        set(gca, 'YTick', linspace(1,size(rMSE_parfor,1),numel(kappa_space)));
        set(gca, 'YTickLabel', kappa_space); xtickangle(45);
        set(gca, 'XTick', linspace(1,size(rMSE_parfor,2),numel(lambda_space)));
        set(gca, 'XTickLabel', lambda_space);
        axis equal tight; title([solvers{s} ', rMSE']);
        [~,min_rMSE_idx] = nanmin(rMSE_parfor(:));
        subplot(224); videosc([n,n],X_parfor(:,:,min_rMSE_idx));
        title(['\lambda = ' num2str(lambda_grid(min_rMSE_idx)) ', '...
               '\kappa = ' num2str(kappa_grid(min_rMSE_idx)) ', '...
               '\mu = ' num2str(mu_grid(min_rMSE_idx)) ', '...
               'rMSE = ' num2str(rMSE_parfor(min_rMSE_idx))]);
        
        figure(fig1);
        for f = 1:sim_param.nbr_frames-1
            axes(ha(f+(1+s)*(sim_param.nbr_frames-1)));
            imagesc(reshape(X_parfor(:,f,min_rMSE_idx),[n,n]),[0,1]); axis equal tight off;
        end
        figure(fig2);
        if i > 1, delete(h2(s)); end
        h2(s) = plot(1:sim_param.nbr_frames-1,rMSE_per_frame(X_parfor(:,:,min_rMSE_idx),X_occ(:,2:end)),[markers{s} '-'],'color',cord(s,:),'LineWidth',2);
        legend(h2,solvers(1:s),'Location','Best','Interpreter','LaTeX'); ylabel('rMSE'); xlabel('Frame index');
        figure(fig3);
        if i > 1, delete(h3(s)); end
        h3(s) = plot(1:sim_param.nbr_frames-1,F1_per_frame(X_parfor(:,:,min_rMSE_idx),X_occ(:,2:end)),[markers{s} '-'],'color',cord(s,:),'LineWidth',2);
        legend(h3,solvers(1:s),'Location','Best','Interpreter','LaTeX'); ylabel('F1 score'); xlabel('Frame index');

        drawnow;
    end
    
    % Save
    results(s).lambda   = lambda_grid;
    results(s).kappa    = kappa_grid;
    results(s).mu       = mu_grid;
    results(s).solver   = solvers{s};
    results(s).Xhat     = X_parfor;
    results(s).rMSE     = rMSE_parfor;
    [~,results(s).min_idx] = nanmin(rMSE_parfor(:));

    ts_disp([solvers{s} ' search completed.']);
    
%     % Plot combined figure
%     splen = length(solvers) + 3;
%     figure(100); clf;
%     subplot(splen,1,1); videosc([n,n],X_gt + M_occ(:)*ones(1,sim_param.nbr_frames));
%     title('Ground truth (with occlusion)');
%     subplot(splen,1,2); videosc([n,n],X_occ);
%     title('Ground truth (after subtracting occlusion)');
%     for ss = 1:s
%         subplot(splen,1,2+ss); videosc([n,n],results(ss).Xhat(:,:,results(ss).min_idx));
%         title(solvers{ss});
%         subplot(splen,1,splen); hold on;
%         plot(rMSE_per_frame(results(ss).Xhat(:,:,results(ss).min_idx),X_occ(:,2:end)),'o-','LineWidth',2);
%     end
%     legend(solvers); ylabel('rMSE'); xlabel('Frame index');
end

ts_disp('Parameter search completed.');

return;

%% Run monte carlo simulations

NbrTrials = 50;

cord = get(gca,'colororder');
fig3 = figure(103); clf; set(fig3,'Units','normalized','Position',[0.1 0.5 0.4 0.25]); grid on; hold on; drawnow;
fig4 = figure(104); clf; set(fig4,'Units','normalized','Position',[0.1 0.1 0.4 0.25]); grid on; hold on; drawnow;
for s = 1:length(solvers)
    % Configure solver
    [~,~,~,solve_frame] = configure_solver(solvers{s},sim_param);

    % Retrieve optimal parameters
    lambda  = results(s).lambda(results(s).min_idx);
    kappa   = results(s).kappa(results(s).min_idx);
    mu      = results(s).mu(results(s).min_idx);

    % Run trials in parallel
    X_parfor = zeros(size(X_gt,1),size(X_gt,2)-1, NbrTrials);
    parfor t = 1:NbrTrials
        % Simulate scenario
        [~,~,~,X_occ,Y,Phi] = run_simuation(n,t,sim_param);
        % Run solver
        X_parfor(:,:,t) = solve_video(solve_frame,Y,Phi,X_occ(:,1),lambda,kappa,mu,dyn_prediction);
    end
    
    % Aggregate rMSE/F1 per frame results
    results(s).rMSEframes = zeros(size(X_gt,2)-1, NbrTrials);
    results(s).F1frames = zeros(size(X_gt,2)-1, NbrTrials);
    for t = 1:NbrTrials
        [~,~,~,X_occ,~,~] = run_simuation(n,t,sim_param); % Simulate scenario
        results(s).rMSEframes(:,t)  = rMSE_per_frame(X_parfor(:,:,t),X_occ(:,2:end));
        results(s).F1frames(:,t)    = F1_per_frame(X_parfor(:,:,t),X_occ(:,2:end));
    end
    
    % Display average rMSE/F1 per frame
    figure(103); plot(mean(results(s).rMSEframes,2),[markers{s} '-'],'color',cord(s,:),'LineWidth',2);
    legend(solvers,'Location','Best','Interpreter','LaTeX'); ylabel('Aggregated rMSE'); xlabel('Frame index');
    figure(104); plot(mean(results(s).F1frames,2),[markers{s} '-'],'color',cord(s,:),'LineWidth',2);
    legend(solvers,'Location','Best','Interpreter','LaTeX'); ylabel('F1 score'); xlabel('Frame index');
    drawnow;
end

return;

%% Replot complete figure

fontsize = 10;
% disp_frames = 1:sim_param.nbr_frames-1;
disp_frames = [6 7 8 14 15 16];

fig1 = figure(100); clf;
set(fig1,'Units','normalized','Position',[0.1 0.1 0.225 0.4]); clf;
ha = tight_subplot(length(solvers)+2,length(disp_frames),[.01 .01],[.01 .05],[.05 .01]);
for f = 1:length(disp_frames)
    fr = disp_frames(f);
    axes(ha(f)); imagesc(-reshape(X_befocc(:,fr+1),[n,n])); axis image; set(gca,'xticklabel',[],'yticklabel',[]);
    title(['Frame ' num2str(fr)],'Interpreter','LaTeX');
    axes(ha(f+length(disp_frames))); imagesc(reshape(X_occ(:,fr+1),[n,n])); axis image; set(gca,'xticklabel',[],'yticklabel',[]);
end
axes(ha(length(disp_frames)+1)); text(-0.2*n,0.5*n,'Observation','Interpreter','LaTeX','FontSize',fontsize,'Rotation',90,'HorizontalAlignment','center');
fig2 = figure(101); clf;
set(fig2,'Units','normalized','Position',[0.1 0.5 0.25 0.15]);
grid on; hold on;
drawnow;
fig3 = figure(102); clf;
set(fig3,'Units','normalized','Position',[0.1 0.1 0.25 0.15]);
grid on; hold on;
drawnow;
for s = 1:length(solvers)
    % Plot combined figure
    figure(fig1);
    for f = 1:length(disp_frames)
        fr = disp_frames(f);
        axes(ha(f+(1+s)*length(disp_frames)));
        imagesc(reshape(results(s).Xhat(:,fr,results(s).min_idx),[n,n]),[0,1]); axis image; set(gca,'xticklabel',[],'yticklabel',[]);
        if f == 1
        text(-0.2*n,0.5*n,solvers{s},'FontSize',fontsize,'Interpreter','LaTeX','Rotation',90,'HorizontalAlignment','center');    
        end
    end
    figure(fig2);
    plot(1:sim_param.nbr_frames-1,rMSE_per_frame(results(s).Xhat(:,:,results(s).min_idx),X_occ(:,2:end)),[markers{s} '-'],'LineWidth',2);
    legend(solvers,'Location','NorthWest','Interpreter','LaTeX'); ylabel('rMSE','Interpreter','LaTeX'); xlabel('Frame index','Interpreter','LaTeX');
    figure(fig3);
    plot(1:sim_param.nbr_frames-1,F1_per_frame(results(s).Xhat(:,:,results(s).min_idx),X_occ(:,2:end)),[markers{s} '-'],'LineWidth',2);
    ylabel('F1 score','Interpreter','LaTeX'); xlabel('Frame index','Interpreter','LaTeX');
    drawnow;
end
figure(fig1); colormap(flipud(gray));
figure(fig1); saveFig2PDF([filename '_qualitative']);
figure(fig2); saveFig2PDF([filename '_rMSE']);
figure(fig3); saveFig2PDF([filename '_F1']);

%% Simulation and solver settings

function [M_occ,X_gt,X_befocc,X_occ,Y,Phi] = run_simuation(n,rnd_seed,sim_param)
% Create an occlusion mask
M_occ = zeros(sim_param.imsize);
M_occ(max(1,floor( sim_param.imsize(1)*(0.5-sim_param.occl_size(1)/2) )) : ...
      min(n,ceil( sim_param.imsize(1)*(0.5+sim_param.occl_size(1)/2) )) , ...
      max(1,floor( sim_param.imsize(2)*(0.5-sim_param.occl_size(2)/2) )) : ...
      min(n,ceil( sim_param.imsize(2)*(0.5+sim_param.occl_size(2)/2) )) ) = -2;

rng(rnd_seed,'Twister'); [~,~,X_gt] = simulator(sim_param);

% Apply occlusion
X_befocc = X_gt + M_occ(:)*ones(1,sim_param.nbr_frames);
X_occ = max(X_befocc,0);

switch sim_param.meas_method
case 'iden'
    Phi = zeros(n^2,n^2,sim_param.nbr_frames);
    for f = 1:sim_param.nbr_frames, Phi(:,:,f) = speye(n^2); end
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

function [lambda_space,kappa_space,mu_space,solve_frame] = configure_solver(solver,sim_param)
switch solver
case 'L1'
    lambda_space = logspace(-3,-1,5);
    kappa_space = logspace(-4,-1,5);
    mu_space = 0;
    solve_frame = @(y,A,x0,lambda,kappa,mu) solver_BPDN_L1DF(sim_param.imsize,y,A,x0,lambda,kappa);
case 'RWL1'
    lambda_space = logspace(-3,-1,5);
    kappa_space = logspace(-1,1,3);
    mu_space = 0;
    solve_frame = @(y,A,x0,lambda,kappa,mu) solver_RWL1DF(sim_param.imsize,y,A,x0,lambda,kappa);
case 'BOT'
    lambda_space = logspace(-3,-1,7);
    kappa_space = logspace(-3,-1,7);
    mu_space = 0;
    solve_frame = @(y,A,x0,lambda,kappa,mu) solver_BOTDF_ADMM(sim_param.imsize,y,A,x0,lambda,kappa);
case 'UOT'
    lambda_space = logspace(-3,-1,7);
    kappa_space = logspace(-3,-1,7);
    mu_space = 2;
    solve_frame = @(y,A,x0,lambda,kappa,mu) solver_UOTDF_ADMM(sim_param.imsize,y,A,x0,lambda,kappa,mu);
end
end

%% Solvers

function F1 = F1_per_frame(X_hat,X_gt)
nbr_frames = size(X_hat,2);
F1 = nan(nbr_frames,1);
for f = 1:nbr_frames
    F1(f) = compute_F1( X_hat(:,f) , X_gt(:,f) , 0.01 );
end
end

function rMSE = rMSE_per_frame(X_hat,X_gt)
nbr_frames = size(X_hat,2);
rMSE = nan(nbr_frames,1);
for f = 1:nbr_frames
    rMSE(f) = norm(X_hat(:,f)-X_gt(:,f))^2 / norm(X_gt(:,f))^2;
end
end

function videosc(imsize,X_hat)
nbr_frames = size(X_hat,2);
F = zeros(imsize(1),imsize(2)*nbr_frames);
for f = 1:nbr_frames
    F(:,(f-1)*imsize(2)+1:f*imsize(2)) = reshape(X_hat(:,f),imsize);
end
imagesc(F); axis equal tight off; hold on;
for f = 1:nbr_frames-1, plot((f*imsize(2)+0.5)*ones(1,2),[0,imsize(1)]+0.5,'k'); end
end

function X_hat = solve_video(solve_frame,Y,Phi,Xstar,lambda,kappa,mu,dyn_prediction)
nbr_frames = size(Y,2);
X_hat = zeros(size(Phi,2),nbr_frames-1);
for t = 2:nbr_frames
    y = Y(:,t);
    A = squeeze(Phi(:,:,t));
    switch dyn_prediction
    case 'ground_truth'
        x0 = Xstar(:,t-1);
    case 'previous_estimate'
        if t == 2, x0 = Xstar(:,t-1);
        else,      x0 = X_hat(:,t-2);
        end
    end
    X_hat(:,t-1) = solve_frame(y,A,x0,lambda,kappa,mu);
end
end

function x = solver_BPDN_L1DF(imsize,y,Phi,x0,lambda,kappa)
[~,N] = size(Phi);
x0 = vec(imgaussfilt(reshape(x0,imsize),1));
cvx_begin quiet
cvx_solver SDPT3
variable x(N,1) nonnegative;
minimize( 0.5 * sum_square(vec(y)-Phi*x) ...
          + lambda * norm(x,1) ...
          + kappa * norm(x-x0,1) );
cvx_end
end

function x = solver_RWL1DF(imsize,y,Phi,x0,lambda,kappa)
[~,N] = size(Phi);
x0 = vec(imgaussfilt(reshape(x0,imsize),1));
EPS = 1e-2;
for r = 1:3
if r == 1, w = ones(N,1);
else,      w = 1./(kappa*abs(x0) + abs(x) + EPS);
end
cvx_begin quiet
cvx_solver SDPT3
variable x(N,1) nonnegative;
minimize( 0.5 * sum_square(vec(y)-Phi*x) ...
          + lambda * norm(diag(w)*x,1) );
cvx_end
end
end

function x = solver_BOTDF_ADMM(imsize,y,Phi,x0,lambda,kappa)
opts.rho = 0.1;
opts.maxiter = 2000;
opts.tolerance = 1e-3;
opts.beck_tau1 = 0.1;
opts.beck_tau2 = 1.0;
opts.beck_maxiter = 1;
x = solver_BPDN_BOT_Beckmann_ADMM(imsize,y,Phi,x0,lambda,kappa,opts);
end

function x = solver_UOTDF_ADMM(imsize,y,Phi,x0,lambda,kappa,mu)
opts.rho = 0.1;
opts.maxiter = 2000;
opts.tolerance = 1e-3;
opts.beck_tau1 = 0.1;
opts.beck_tau2 = 1.0;
opts.beck_maxiter = 3;
x = solver_BPDN_UOT_Beckmann_ADMM(imsize,y,Phi,x0,lambda,kappa,mu,opts);
end