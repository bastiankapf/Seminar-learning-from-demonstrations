%% 1. Parameters & letter list
letters = {'B', 'a', 's', 't', 'i'};
letter_states = [6, 4, 4, 4, 3]; % GMM states per letter

T_letter = 100;
in_idx = 1; out_idx = [2, 3];

spacing = 5; % Spacing between letters in the word
current_x_offset = 0;

figure; hold on; grid on;

%% 2. Loop over all letters
target_height = 120; % Target height for the letters in the word

for k = 1:length(letters)
    letter = letters{k};
    nbStates = letter_states(k);
    
    % --- A. Load ---
    file_path = sprintf('%s.mat', letter);
    if ~exist(file_path, 'file')
        file_path = sprintf('/MATLAB Drive/data/2Dletters/%s.mat', letter);
    end
    load(file_path); % Loads 'demos'
    
    nbDemos = length(demos);
    Data = [];
    
    % --- B. Resampling ---
    for n = 1:nbDemos
        pos = demos{n}.pos;
        
        dx = diff(pos(1,:)); dy = diff(pos(2,:));
        s = [0, cumsum(sqrt(dx.^2 + dy.^2))];
        s_target = linspace(0, s(end), T_letter);
        
        x_interp = interp1(s, pos(1,:), s_target, 'pchip');
        y_interp = interp1(s, pos(2,:), s_target, 'pchip');
        
        Data = [Data, [1:T_letter; x_interp; y_interp]];
    end
    
    % --- C. GMM training & GMR regression ---
    [Priors, Mu, Sigma] = init_GMM_time(Data, nbStates);
    [Priors, Mu, Sigma] = EM_GMM(Data, Priors, Mu, Sigma);
    
    t_query = 1:T_letter;
    expData = GMR(Priors, Mu, Sigma, t_query, in_idx, out_idx);
    
    % --- D. Scaling & vertical alignment (baseline alignment) ---
    
    % 1. Keep the scaling (adjust the height)
    current_h = max(expData(2, :)) - min(expData(2, :));
    if current_h < 1, current_h = 1; end
    scale_factor = target_height / current_h;
    
    if ismember(letter, {'a', 's'})
        scale_factor = scale_factor * 0.55; % Lowercase scaling
    elseif ismember(letter, {'i'})
        scale_factor = scale_factor * 0.75; 
    end
    
    % Scale the data
    expData = expData * scale_factor;
    Mu(out_idx, :) = Mu(out_idx, :) * scale_factor;
    Sigma(out_idx, out_idx, :) = Sigma(out_idx, out_idx, :) * (scale_factor^2);
    
    % 2. X alignment (set the left edge to 0)
    min_x = min(expData(1, :));
    expData(1, :) = expData(1, :) - min_x;
    Mu(out_idx(1), :) = Mu(out_idx(1), :) - min_x;
    
    % 3. Y alignment to the baseline
    if k == 1
        baseline_y = min(expData(2, :));
    end
    
    current_min_y = min(expData(2, :));
    shift_y = baseline_y - current_min_y;
    
    expData(2, :) = expData(2, :) + shift_y;
    Mu(out_idx(2), :) = Mu(out_idx(2), :) + shift_y;
    
    % 4. Apply horizontal offset for the word
    expData(1, :) = expData(1, :) + current_x_offset;
    Mu_shifted = Mu;
    Mu_shifted(out_idx(1), :) = Mu(out_idx(1), :) + current_x_offset;
    
    % --- E. Draw the letter ---
    step_dists = sqrt(diff(expData(1,:)).^2 + diff(expData(2,:)).^2);
    jump_idx = find(step_dists > mean(step_dists) * 2.8, 1);
    
    if ~isempty(jump_idx)
        % Split letter (t, i, or a)
        plot(expData(1, 1:jump_idx), expData(2, 1:jump_idx), 'r-', 'LineWidth', 2.5, 'HandleVisibility', 'off');
        plot(expData(1, jump_idx+1:end), expData(2, jump_idx+1:end), 'r-', 'LineWidth', 2.5, 'HandleVisibility', 'off');
        plot([expData(1, jump_idx), expData(1, jump_idx+1)], ...
             [expData(2, jump_idx), expData(2, jump_idx+1)], 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
    else
        plot(expData(1, :), expData(2, :), 'r-', 'LineWidth', 2.5, 'HandleVisibility', 'off');
    end
    
    % Plot GMM ellipses
    plot_GMM_ellipses(Mu_shifted(out_idx,:), Sigma(out_idx,out_idx,:), [0.3 0.8 0.3], 0.25);
    
    % --- F. Offset for the next letter ---
    max_x = max(expData(1, :));
    current_x_offset = max_x + spacing;
end

% Dummy plot for a clean legend
plot(nan, nan, 'r-', 'LineWidth', 2.5, 'DisplayName', 'GMR trajectory');
plot(nan, nan, 'Square', 'Color', [0.3 0.8 0.3], 'MarkerFaceColor', [0.3 0.8 0.3], 'DisplayName', 'GMM Gaussians');

title('GMM/GMR: "Basti"');
xlabel('X position'); ylabel('Y position');
axis equal;
legend show;

%% --- Hilfsfunktionen ---

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
        Sigma(:,:,i) = cov_i + eye(nbVar) * 0.5;
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
