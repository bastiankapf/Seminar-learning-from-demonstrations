%% 1. Load data
load('/MATLAB Drive/data/2Dletters/B.mat');

nbDemos = length(demos);
T = 100; % Target timesteps per demonstration
Data = [];

for n = 1:nbDemos
    pos = demos{n}.pos;
    
    % Cumulative arc-length 
    dx = diff(pos(1,:)); 
    dy = diff(pos(2,:));
    s = [0, cumsum(sqrt(dx.^2 + dy.^2))];
    
    % Resample evenly along the path length s
    s_target = linspace(0, s(end), T);
    x_interp = interp1(s, pos(1,:), s_target, 'pchip');
    y_interp = interp1(s, pos(2,:), s_target, 'pchip');
    
    % Time index t = 1..T
    Data = [Data, [1:T; x_interp; y_interp]];
end

%% 2. GMM parameters & EM algorithm
nbStates = 6; 

in_idx  = 1;      % Time t
out_idx = [2, 3]; % Position [X, Y]

[Priors, Mu, Sigma] = init_GMM_time(Data, nbStates);
[Priors, Mu, Sigma] = EM_GMM(Data, Priors, Mu, Sigma);

disp('GMM trained successfully!');

%% 3. GMR reconstruction
t_query = 1:T; 
expData = GMR(Priors, Mu, Sigma, t_query, in_idx, out_idx);

%% 4. Visualize results
figure; hold on; grid on;

% 1. Original demonstrations
for n = 1:nbDemos
    plot(demos{n}.pos(1,:), demos{n}.pos(2,:), 'Color', [0.75 0.75 0.75], 'LineWidth', 1);
end

% 2. Gaussian components (ellipses)
plot_GMM_ellipses(Mu(out_idx,:), Sigma(out_idx,out_idx,:), [0.2 0.8 0.2], 0.4);

% 3. GMR trajectory
plot(expData(1,:), expData(2,:), 'r-', 'LineWidth', 3, 'DisplayName', 'GMR Trajectory');

title('GMM/GMR Modeling');
xlabel('X position'); ylabel('Y position');
axis equal;
legend show;




%% --- Helper functions (unchanged functions) ---

function [Priors, Mu, Sigma] = init_GMM_time(Data, nbStates)
    [nbVar, nbData] = size(Data);
    Priors = ones(1, nbStates) / nbStates;
    [~, idx] = sort(Data(1,:));
    DataSorted = Data(:, idx);
    diagBlock = floor(nbData / nbStates);
    Mu = zeros(nbVar, nbStates);
    Sigma = zeros(nbVar, nbVar, nbStates);
    
    for i = 1:nbStates
        id = (i-1)*diagBlock + 1 : min(i*diagBlock, nbData);
        Mu(:,i) = mean(DataSorted(:,id), 2);
        cov_i = cov(DataSorted(:,id)');
        Sigma(:,:,i) = cov_i + eye(nbVar) * 0.5; % For 'i', a smaller covariance regularization is enough
    end
end

function [Priors, Mu, Sigma] = EM_GMM(Data, Priors, Mu, Sigma)
    [nbVar, nbData] = size(Data);
    nbStates = size(Mu, 2);
    L = zeros(nbStates, nbData);
    min_cov = eye(nbVar) * 0.2; 
    
    for iter = 1:30
        for i = 1:nbStates
            L(i,:) = Priors(i) * gaussPDF(Data, Mu(:,i), Sigma(:,:,i));
        end
        GAMMA = L ./ (sum(L,1) + realmin);
        
        for i = 1:nbStates
            gamma_sum = sum(GAMMA(i,:));
            Priors(i) = gamma_sum / nbData;
            Mu(:,i) = Data * GAMMA(i,:)' / gamma_sum;
            diff = Data - repmat(Mu(:,i), 1, nbData);
            Sigma_new = (diff * diag(GAMMA(i,:)) * diff') / gamma_sum;
            Sigma(:,:,i) = Sigma_new + min_cov;
        end
    end
end

function prob = gaussPDF(Data, Mu, Sigma)
    [nbVar, nbData] = size(Data);
    diff = Data - repmat(Mu, 1, nbData);
    prob = sum((diff' / Sigma) .* diff', 2);
    prob = exp(-0.5 * prob) / sqrt((2*pi)^nbVar * (det(Sigma) + realmin));
    prob = prob';
end

function expData = GMR(Priors, Mu, Sigma, t_query, in_idx, out_idx)
    nbData = length(t_query);
    nbStates = size(Mu, 2);
    nbVarOut = length(out_idx);
    expData = zeros(nbVarOut, nbData);
    for t = 1:nbData
        x = t_query(t);
        H = zeros(1, nbStates);
        for i = 1:nbStates
            H(i) = Priors(i) * gaussPDF(x, Mu(in_idx,i), Sigma(in_idx,in_idx,i));
        end
        H = H / (sum(H) + realmin);
        for i = 1:nbStates
            mu_k = Mu(out_idx,i) + Sigma(out_idx,in_idx,i) / Sigma(in_idx,in_idx,i) * (x - Mu(in_idx,i));
            expData(:,t) = expData(:,t) + H(i) * mu_k;
        end
    end
end

function plot_GMM_ellipses(Mu, Sigma, color, alpha_val)
    t = linspace(0, 2*pi, 50);
    for k = 1:size(Mu, 2)
        [V, D] = eig(Sigma(:,:,k));
        u = [cos(t); sin(t)];
        pts = V * sqrt(D) * u + Mu(:,k);
        fill(pts(1,:), pts(2,:), color, 'FaceAlpha', alpha_val, 'EdgeColor', 'none');
    end
end
