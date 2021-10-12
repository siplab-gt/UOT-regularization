function [x] = simulate_pixels(imsize,F,K,B,magnitude,mass_growth_profile,mass_growth_rate)

% n = frame width
% F = number of frames
% K = number of targets
% B = distance that pixels can move
% magnitude = 'pos', 'neg', 'real' (default = 'pos')
% mass_growth_profile = 'static', 'ramp', 'sine'
% mass_growth_rate = [0,1] i.e., 0 = static, 1 = very dynamic

m = imsize(1); n = imsize(2);
N = m*n;
[rr,cc] = meshgrid(1:n,1:m);
% N = n^2;
% [rr,cc] = meshgrid(1:n);
if nargin < 5, magnitude = 'pos'; end

% Special case: K = N (paint everything 1)
if K == N, x = ones(N,F); return; end

% Generate mass_growth_profile
if nargin < 6, mass_growth_profile = 'static'; end
if nargin < 7, mass_growth_rate = 0; end
mass_growth_rate = max(0,min(2-1e-5,mass_growth_rate));
switch lower(mass_growth_profile)
case 'static'
    mass_profile = ones(F,1);
case 'ramp'
    mass_profile = 1 + linspace(-mass_growth_rate/2,mass_growth_rate/2,F);
case 'sine'
    mass_profile = 1 + mass_growth_rate/2*sin(((1:F)-1)/F*2*pi);
case 'tri'
    mass_profile = generate_bounded_profile(F,mass_growth_rate);
    % mass_profile = 1-mass_growth_rate/2 + mass_growth_rate*tripuls((1:F)-mean(1:F),F-1);
end

for attempt = 1:10000 % max attempts
    
    % Reset
    empty_support_restart = 0;
    x = zeros(N,F);

    % Generate first frame
    switch lower(magnitude)
    case 'pos' % strictly +1s
        x(randperm(N,K)',1) = mass_profile(1);
    case 'neg' % strictly -1s
        x(randperm(N,K)',1) = -mass_profile(1);
    case 'real' % random sign of 1s
        x(randperm(N,K)',1) = sign(randn(K,1))*mass_profile(1);
    end
 
    % Generate subsequent frames
    for f = 2:F
        targets = find(abs(x(:,f-1))>0);
        for t = 1:K
            % Identify each target
            [tar_x,tar_y] = ind2sub([m,n],targets(t));
            
            % Generate matrix w.r.t. distance from this target
            S = sqrt((rr-tar_y).^2+(cc-tar_x).^2);
            % Generate a neg exponential probabilities based on mean B
            S = exp(-(B-S).^2/(0.25)); % spread of 0.25 (hard coded)
            % Remove exisiting targets from support
            S = S - S.*reshape(x(:,f)>0,m,n);
            % Remove prev location from support
            % S(tar_x,tar_y) = 0;
            % Normalize
            S = S / sum(S(:));
            % Reset if no empty locations left
            new_sup = find(S > 0);
            if isempty(new_sup), empty_support_restart = 1; break; end
            % Place target randomly in new support
            new_loc = randsample(1:n*m,1,true,S(:));
            % Assign magnitude
            x(new_loc,f) = mass_profile(f); % magnitude based on profile
            
%             % Find support for each target
%             S = sqrt((rr-tar_y).^2+(cc-tar_x).^2)<=B;
%             % Remove existing targets from support
%             S = max( S-reshape(x(:,f)>0,m,n) , 0 );
%             % Reset if no empty locations left
%             new_sup = find(S == 1);
%             % Force target to move from original location
%             new_sup = setdiff(new_sup,targets(t));
%             if isempty(new_sup), empty_support_restart = 1; break; end
%             % Place target randomly in new support
%             new_loc = new_sup(randperm(length(new_sup),1));
%             % Assign magnitude
%             % x(new_loc,f) = x(targets(t),f-1); % unchanged magnitude
%             x(new_loc,f) = mass_profile(f); % magnitude based on profile
            
        end
        if empty_support_restart == 1, break; end 
    end
    if empty_support_restart == 0, break; end % Terminate upon success
end

if empty_support_restart == 1, error(['Too dense! Attempts =' num2str(attempt)]); end

end

function profile = generate_bounded_profile(len,rate)
profile = zeros(len,1);
rate = min(1,max(0,rate)); % bound to [0,1]
nbr_steps = floor(1/rate) + 1;
step_val = (0:nbr_steps-1)*rate;
direction = 1; % 1 = pos, -1 = neg
step_idx = randperm(nbr_steps,1); % randomize phase offset
profile(1) = step_val(step_idx);
for i = 2:len
    next_step_idx = step_idx + direction;
    if (next_step_idx > nbr_steps) || (next_step_idx < 1)
        direction = -direction;
    end
    step_idx = step_idx + direction;
    profile(i) = step_val(step_idx);
end
bound = max(profile) - min(profile);
profile = 1 + profile - bound/2;
end
