function [Y,Phi,S_gt,L_gt] = simulator(param)
% Parameters
imsize              = param.imsize;                 % imsize(1) = height, imsize(2) = width
nbr_frames          = param.nbr_frames;             % total number of frames in video
meas_method         = param.meas_method;            % 'cs'   - compressive sensing matrix (default)
                                                    % 'iden' - denoising (identity matrix)
K                   = param.K;                      % number of active pixels (K<imsize(1)*imsize(2))
B                   = param.B;                      % speed of motion between frames
M                   = param.M;                      % number of measurements
R                   = param.R;                      % rank of clutter
Q                   = param.Q;                      % percent of targets to dissappear
noise_sigma         = param.noise_sigma;            % noise std dev
magnitude           = param.magnitude;              % 'pos'  - positive only values (default)
                                                    % 'neg'  - negative only values 
                                                    % 'real' - pos and neg values.
mass_growth_profile = param.mass_growth_profile;    % 'static' - constant ones
                                                    % 'ramp' - from down to up
                                                    % 'sine' - 1 sine wave cycle
                                                    % 'tri'  - up then down
mass_growth_rate    = param.mass_growth_rate;       % between 0 and 1

N = imsize(1)*imsize(2);

% Targets (ground truth)
S_gt = simulate_pixels(imsize, nbr_frames, ceil(K*N), B, magnitude, mass_growth_profile, mass_growth_rate);

% % Randomly remove targets
% tar_idx = find(S_gt);
% tar_rm_idx = tar_idx(randperm(length(tar_idx),ceil(0.05*length(tar_idx))));
% S_gt(tar_rm_idx) = 0;

% Clutter (ground truth)
rank = ceil(R*nbr_frames);
L_gt = rand(size(S_gt,1),rank) * rand(rank,size(S_gt,2));
L_gt = L_gt / rank; % normalize by rank
L_gt = L_gt / 0.25; % make mean 1

% Random measurements
switch lower(meas_method)
case 'cs'
    [Y, Phi] = take_gaussian_meas(S_gt + L_gt, ceil(M*N), noise_sigma^2);
case 'iden'
    Phi = zeros(N,N,nbr_frames);
    for f = 1:nbr_frames, Phi(:,:,f) = speye(N); end
    Y = S_gt + L_gt + noise_sigma*randn(size(S_gt));
otherwise
    Phi = [];
    Y = S_gt + L_gt + noise_sigma*randn(size(S_gt));
end
% Phi_blk = mat2cell(Phi,size(Phi,1),size(Phi,2),ones(1,size(Phi,3))); Phi_blk = blkdiag(Phi_blk{:});

end