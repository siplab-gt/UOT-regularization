switch filename
case 'Figure001'
    noise_sigma_space = logspace(log10(0.005),log10(0.5),10);
    var_name = 'Noise (\sigma)';
case 'Figure002'
    M_space = linspace(0.4,0.8,10);
    var_name = 'Compression ratio (M/N)';
case 'Figure003'
    K_space = linspace(0.01,0.30,10);
    var_name = 'Sparsity ratio (K/N)';
case 'Figure004'
    R_space = linspace(1/sim_param.nbr_frames,1.0,sim_param.nbr_frames);
    var_name = 'Rank ratio (R/N)';
case 'Figure005'
    nbr_frames_space = 2:9;
    var_name = 'Batch size';
end