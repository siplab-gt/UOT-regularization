function param = get_solver_param_via_search(solvers,search_seeds,F1_threshold,eval_weights,sim_param,ADMM_opts)

% Parameters
prob_sz = [sim_param.imsize(1)*sim_param.imsize(2),sim_param.nbr_frames];
lambda_init = 0.1; kappa_init  = 1e-2;

% Pattern Search parameters
options = optimoptions('patternsearch',...
                       ...'Display','iter',...
                       'MaxFunctionEvaluations',50);
                   
for s = 1:length(solvers)
disp(['(' datestr(now,'HH:MM:SS.FFF') ') Param Search on ' solvers{s} ' started.']);
switch solvers{s}
case 'RPCA'
    % Try exhaustive search
    tic; [p,fval] = exhaustive_grid_search(solvers{s},sim_param,search_seeds,F1_threshold,eval_weights,ADMM_opts);
    param(s).lambda = p.lambda;
    param(s).gamma  = p.gamma;
    param(s).kappa  = [];
    param(s).mu     = [];
    ts_disp(['Exhaustive Search on ' solvers{s} ' took ' num2str(toc) 's : ' ...
             'best metric = ' num2str(fval) ', '...
             'lambda = ' num2str(param(s).lambda) ', '...
             'gamma = ' num2str(param(s).gamma) ]);
case {'RPCA+L1-DF','RPCA+BOT-DF'}
    % Try exhaustive search
    tic; [p,fval] = exhaustive_grid_search(solvers{s},sim_param,search_seeds,F1_threshold,eval_weights,ADMM_opts);
    param(s).lambda = p.lambda;
    param(s).gamma  = p.gamma;
    param(s).kappa  = p.kappa;
    param(s).mu     = [];
    ts_disp(['Exhaustive Search on ' solvers{s} ' took ' num2str(toc) 's : ' ...
             'best metric = ' num2str(fval) ', '...
             'lambda = ' num2str(param(s).lambda) ', '...
             'gamma = ' num2str(param(s).gamma) ', '...
             'kappa = ' num2str(param(s).kappa) ]);
case 'RPCA+UOT-DF'
    % Try exhaustive search
    tic; [p,fval] = exhaustive_grid_search(solvers{s},sim_param,search_seeds,F1_threshold,eval_weights,ADMM_opts);
    param(s).lambda = p.lambda;
    param(s).gamma  = p.gamma;
    param(s).kappa  = p.kappa;
    param(s).mu     = p.mu;
    ts_disp(['Exhaustive Search on ' solvers{s} ' took ' num2str(toc) 's : ' ...
             'best metric = ' num2str(fval) ', '...
             'lambda = ' num2str(param(s).lambda) ', '...
             'gamma = ' num2str(param(s).gamma) ', '...
             'kappa = ' num2str(param(s).kappa) ', '...
             'mu = ' num2str(param(s).mu) ]);
end
end
end

function [param,fval] = exhaustive_grid_search(solver,sim_param,search_seeds,F1_threshold,eval_weights,ADMM_opts)

search_param.display_plots  = 0; %%%%%% <-- For Diagnostic Display %%%%%%
search_param.nbr_iter       = 3;

search_param.lambda_center      = -2.5;
search_param.lambda_width       = 5;
search_param.lambda_grid_size   = 5;
search_param.kappa_center       = -3.5;
search_param.kappa_width        = 5;
search_param.kappa_grid_size    = 5;
search_param.mu_center          = 0;
search_param.mu_width           = 0;
search_param.mu_grid_size       = 1;

switch solver
case 'RPCA'
    search_param.lambda_center      = -3;
    search_param.lambda_width       = 6;
    search_param.lambda_grid_size   = 25;
    search_param.kappa_grid_size    = 1;
    [param.lambda,param.gamma,~,~,fval] = ...
        logspace_pattern_search(solver,sim_param,search_seeds,search_param,F1_threshold,eval_weights,ADMM_opts);
case {'RPCA+L1-DF','RPCA+BOT-DF'}
    search_param.lambda_grid_size   = 5;
    search_param.kappa_grid_size    = 5;
    [param.lambda,param.gamma,param.kappa,~,fval] = ...
        logspace_pattern_search(solver,sim_param,search_seeds,search_param,F1_threshold,eval_weights,ADMM_opts);
case 'RPCA+UOT-DF'
    lambda = 10^(-2.5);
    kappa  = 10^(-3.5);
    mu     = 1;
    for i = 1:2
        % Stage 1: Fix mu, search lambda/kappa
        search_param.lambda_center      = log10(lambda);
        search_param.lambda_width       = 5/(2^(i-1));
        search_param.lambda_grid_size   = 5;
        search_param.kappa_center       = log10(kappa);
        search_param.kappa_width        = 5/(2^(i-1));
        search_param.mu_center          = log10(mu);
        search_param.mu_width           = 0;
        search_param.mu_grid_size       = 1;
        [lambda, gamma, kappa, mu, fval] = ...
            logspace_pattern_search(solver,sim_param,search_seeds,search_param,F1_threshold,eval_weights,ADMM_opts);
        ts_disp(['Param search on ' solver ' (Stage 1: find lambda/kappa) : ' ...
                 'best metric = ' num2str(fval) ', '...
                 'lambda = ' num2str(lambda) ', '...
                 'gamma = ' num2str(gamma) ', '...
                 'kappa = ' num2str(kappa) ', '...
                 'mu = ' num2str(mu) ]);

        % Stage 2: Fix lambda, search kappa/mu
        search_param.lambda_center      = log10(lambda); % search at found lambda
        search_param.lambda_width       = 0;
        search_param.lambda_grid_size   = 1; 
        search_param.kappa_center       = log10(kappa); % search around found kappa
        search_param.kappa_width        = 2/(2^(i-1));
        search_param.mu_center          = log10(mu); % search between 0.1 and 10
        search_param.mu_width           = 2/(2^(i-1));
        search_param.mu_grid_size       = 5;
        [lambda, gamma, kappa, mu, fval] = ...
            logspace_pattern_search(solver,sim_param,search_seeds,search_param,F1_threshold,eval_weights,ADMM_opts);
        ts_disp(['Param search on ' solver ' (Stage 2: find mu/kappa) : ' ...
                 'best metric = ' num2str(fval) ', '...
                 'lambda = ' num2str(lambda) ', '...
                 'gamma = ' num2str(gamma) ', '...
                 'kappa = ' num2str(kappa) ', '...
                 'mu = ' num2str(mu) ]);
    end    
    
    [fval,fidx] = min(fval);
    param.lambda = lambda(fidx);
    param.gamma  = gamma(fidx);
    param.kappa  = kappa(fidx);
    param.mu     = mu(fidx);
end
end

function [lambda,gamma,kappa,mu,fval] = logspace_pattern_search(solver,sim_param,search_seeds,search_param,F1_threshold,eval_weights,ADMM_opts)
search_param.space_reduc    = 0.5; % HARD CODE
lambda_center       = search_param.lambda_center;
lambda_width        = search_param.lambda_width;
lambda_grid_size    = search_param.lambda_grid_size;
kappa_center        = search_param.kappa_center;
kappa_width         = search_param.kappa_width;
kappa_grid_size     = search_param.kappa_grid_size;
mu_center        	= search_param.mu_center;
mu_width            = search_param.mu_width;
mu_grid_size        = search_param.mu_grid_size;
nbr_iter            = search_param.nbr_iter;
space_reduc         = search_param.space_reduc;
display_plots       = search_param.display_plots;
for j = 1:nbr_iter
    lambda_space = logspace(lambda_center-lambda_width*space_reduc, lambda_center+lambda_width*space_reduc, lambda_grid_size);
    kappa_space  = logspace(kappa_center-kappa_width*space_reduc,   kappa_center+kappa_width*space_reduc,   kappa_grid_size);
    mu_space     = logspace(mu_center-mu_width*space_reduc,         mu_center+mu_width*space_reduc,         mu_grid_size);
    [lambda_grid,kappa_grid,mu_grid] = meshgrid(lambda_space,kappa_space,mu_space);
    
    f = zeros(numel(lambda_grid),1);
    parfor i = 1:numel(f)
        f(i) = evaluate_solver(solver,sim_param,search_seeds,lambda_grid(i),kappa_grid(i),mu_grid(i),F1_threshold,eval_weights,ADMM_opts);
    end
    [fval,fidx] = min(f);
    
    if display_plots
        figure;
        if kappa_grid_size == 1 && mu_grid_size == 1
            semilogx(lambda_space,f); hold on; plot(lambda_grid(fidx),fval,'ro','Markersize',12); xlabel('\lambda');
        end
        if kappa_grid_size > 1 && mu_grid_size == 1
            imagesc(reshape(log10(f),length(kappa_space),length(lambda_space))); axis equal tight;
            [x,y]  = ind2sub([length(lambda_space),length(kappa_space)],fidx);
            hold on; plot(y,x,'ro','Markersize',12);
            xlabel('\lambda'); xticks(1:length(lambda_space)); xticklabels(mat2cell(num2str(lambda_space(:)),length(lambda_space))); xtickangle(45);
            ylabel('\kappa');  yticks(1:length(kappa_space));  yticklabels(mat2cell(num2str(kappa_space(:)),length(kappa_space)));
        end
        if lambda_grid_size == 1 && kappa_grid_size > 1 && mu_grid_size > 1
            imagesc(reshape(log10(f),length(mu_space),length(kappa_space))); axis equal tight;
            [x,y]  = ind2sub([length(kappa_space),length(mu_space)],fidx);
            hold on; plot(y,x,'ro','Markersize',12);
            xlabel('\mu');  xticks(1:length(mu_space));  xticklabels(mat2cell(num2str(mu_space(:)),length(mu_space)));
            ylabel('\kappa'); yticks(1:length(kappa_space)); yticklabels(mat2cell(num2str(kappa_space(:)),length(kappa_space))); xtickangle(45);
        end
        title(['Iter ' num2str(j) ', Exhaustive Search on ' solver ' (min = ' num2str(fval) ')']); drawnow;
    end
    
    lambda_center = log10(lambda_grid(fidx)); lambda_width = lambda_width*space_reduc;
    kappa_center  = log10(kappa_grid(fidx));  kappa_width  = kappa_width*space_reduc;
    mu_center     = log10(mu_grid(fidx));     mu_width     = mu_width*space_reduc;
end
lambda = lambda_grid(fidx);
gamma  = lambda_grid(fidx) * sqrt(max(sim_param.imsize(1)*sim_param.imsize(2),sim_param.nbr_frames));
kappa  = kappa_grid(fidx);
mu     = mu_grid(fidx);
end

function fval = evaluate_solver(solver,sim_param,search_seeds,lambda,kappa,mu,F1_threshold,eval_weights,ADMM_opts)
gamma = lambda * sqrt(max(sim_param.imsize(1)*sim_param.imsize(2),sim_param.nbr_frames));
S_rMSE = zeros(length(search_seeds),1);
L_rMSE = zeros(length(search_seeds),1);
F1     = zeros(length(search_seeds),1);
for i = 1:length(search_seeds)
    rng(search_seeds(i),'twister'); [Y,Phi,S_gt,L_gt] = simulator(sim_param);
    switch solver
    case 'RPCA'
        [S,L] = solver_RPCA_ADMM(Y,Phi,lambda,gamma,ADMM_opts);
    case 'RPCA+L1-DF'
        [S,L] = solver_RPCA_ConvL1DF_ADMM(sim_param.imsize,Y,Phi,lambda,gamma,kappa,ADMM_opts);
    case 'RPCA+BOT-DF'
        [S,L] = solver_RPCA_BOT_Beckman_ADMM(sim_param.imsize,Y,Phi,lambda,gamma,kappa,ADMM_opts);
    case 'RPCA+UOT-DF'
        [S,L] = solver_RPCA_UOT_Beckman_ADMM(sim_param.imsize,Y,Phi,lambda,gamma,kappa,mu,ADMM_opts);
    end
    S_rMSE(i) = compute_rMSE(S,S_gt);
    L_rMSE(i) = compute_rMSE(L,L_gt);
    F1(i)     = compute_F1(S,S_gt,F1_threshold);
end

% Performance metric
fval = eval_weights(1)*mean(S_rMSE) + eval_weights(2)*mean(L_rMSE) + eval_weights(3)*mean(1-F1);
end

function val = compute_rMSE(S,S_gt)
val = norm(vec(S-S_gt))^2 / norm(vec(S_gt))^2;
end