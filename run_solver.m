function [S,L,time] = run_solver(solver,imsize,Y,Phi,param,ADMM_opts)
tic
switch solver
case 'RPCA'
    [S,L] = solver_RPCA_ADMM(Y,Phi,param.lambda,param.gamma,ADMM_opts);
case 'RPCA+L1-DF'
      [S,L] = solver_RPCA_ConvL1DF_ADMM(imsize,Y,Phi,param.lambda,param.gamma,param.kappa,ADMM_opts);
case 'RPCA+BOT-DF'
    [S,L] = solver_RPCA_BOT_Beckman_ADMM(imsize,Y,Phi,param.lambda,param.gamma,param.kappa,ADMM_opts);
case 'RPCA+UOT-DF'
    [S,L] = solver_RPCA_UOT_Beckman_ADMM(imsize,Y,Phi,param.lambda,param.gamma,param.kappa,param.mu,ADMM_opts);
end
time = toc;
end