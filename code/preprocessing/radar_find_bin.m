% radar_find_bin.m  Stage 4 -- Select radar bins with strongest cardiac signature per body site.
%
% For each anatomical site (heart, wrist, head, neck), runs the
% autocorrelation-based bin finder, selects the best candidate(s), crops
% the radar data cube around the selected bin, and saves the result.
%
% Requires workspace variables from Stages 1-3:
%   imgs       -- 3-D radar data cube [frames x range x angle]
%   cut_index  -- [start_sample, end_sample] analysis window
%
% Produces:
%   Saves heart_example.mat, wrist_example.mat, head_example.mat,
%   neck_example.mat in the example recording folder.

SAVE_DIR = fullfile('..', '..', 'data', 'example', 'raw', 'example_user', 'example_recording');

% Shared search parameters across body sites
target_bpm     = 114;
thresholdOnBpm = 66;

%% ========================================================================
%  Heart -- autocorrelation bin search
%  ========================================================================
heart_distance       = 0.45;       % expected distance to heart (m)
target_angle         = -40;        % expected azimuth (deg)
angleThresholdDegree = 20;
fr                   = [0.5, 10];  % bandpass range (Hz)
secondDerivativeFlag = 1;

[candidateHistory, ~, fs] = find_bin(imgs, heart_distance, ...
    cut_index(1, 1), cut_index(1, 2), ...
    'thresholdOnBpm', thresholdOnBpm, 'targetBpm', target_bpm, ...
    'secondDerivative', secondDerivativeFlag, 'filter_range', fr, ...
    'targetAngleDegree', target_angle, 'angleThresholdDegree', angleThresholdDegree);

% Keep candidates with non-zero lag, rank by signal magnitude
candidateHistory = candidateHistory(candidateHistory(:, 5) ~= 0, :);
candidateHistory = sortrows(candidateHistory, 2, 'descend');
N = 40;
history_list = candidateHistory(1:min(N, size(candidateHistory, 1)), :);

found_bin = [history_list(1, 3), history_list(1, 4)];

% Crop radar data around the selected heart bin
heart_bin_threshold = 10;
best_range = history_list(1, 3);
best_angle = history_list(1, 4);
if min(128, best_angle + heart_bin_threshold) == 128
    data = imgs(cut_index(1, 1):cut_index(1, 2), best_range, end-heart_bin_threshold*2:end);
else
    data = imgs(cut_index(1, 1):cut_index(1, 2), best_range, ...
        best_angle - heart_bin_threshold:min(128, best_angle + heart_bin_threshold));
end
save(fullfile(SAVE_DIR, 'heart_example'), 'data');

%% ========================================================================
%  Wrist -- lag-consensus bin search
%  ========================================================================
wrist_distance       = 0.10;  % expected distance to wrist (m)
angleThresholdDegree = 20;

[candidateHistory, ~, fs] = find_bin(imgs, wrist_distance, ...
    cut_index(1, 1), cut_index(1, 2), ...
    'angleThresholdDegree', angleThresholdDegree, ...
    'thresholdOnBpm', thresholdOnBpm, 'targetBpm', target_bpm);

candidateHistory = candidateHistory(candidateHistory(:, 5) ~= 0, :);

% Find the lag value that appears most frequently among candidates
threshold = 4;
sorted_lags = sort(candidateHistory(:, 5));
[most_frequent_lag, ~] = most_appear_number(sorted_lags, threshold);

% Keep only candidates whose lag is close to the consensus
consensus_bins = candidateHistory(abs(candidateHistory(:, 5) - most_frequent_lag) <= threshold, :);
history_list   = consensus_bins;

% Pick the most common range bin, then find the angular midpoint
dominant_range = mode(history_list(1:min(10, size(history_list, 1)), 3));
target_list = history_list(1:min(10, size(history_list, 1)), 3:4);
target_list = target_list(target_list(:, 1) == dominant_range, :);
angle_s = min(target_list(:, 2));
angle_l = max(target_list(:, 2));
angle_m = floor(angle_s + (angle_l - angle_s) / 2);

found_bin = [dominant_range, angle_m];

% Crop and save wrist data
wrist_bin_threshold = 10;
data = imgs(cut_index(1, 1):cut_index(1, 2), dominant_range, ...
    angle_m - wrist_bin_threshold:angle_m + wrist_bin_threshold);
save(fullfile(SAVE_DIR, 'wrist_example'), 'data');

%% ========================================================================
%  Head -- spatial midpoint bin search
%  ========================================================================
head_distance        = 0.70;      % expected distance to head (m)
target_angle         = -15;
angleThresholdDegree = 20;
fr                   = [0.7, 4];
secondDerivativeFlag = 0;

[candidateHistory, ~, fs] = find_bin(imgs, head_distance, ...
    cut_index(1, 1), cut_index(1, 2), ...
    'thresholdOnBpm', thresholdOnBpm, 'targetBpm', target_bpm, ...
    'targetAngleDegree', target_angle, 'angleThresholdDegree', angleThresholdDegree, ...
    'filter_range', fr, 'secondDerivative', secondDerivativeFlag);

candidateHistory = candidateHistory(candidateHistory(:, 5) ~= 0, :);

% Retry with second derivative if no candidates found
if isempty(candidateHistory)
    secondDerivativeFlag = 1;
    [candidateHistory, ~, fs] = find_bin(imgs, head_distance, ...
        cut_index(1, 1), cut_index(1, 2), ...
        'thresholdOnBpm', thresholdOnBpm, 'targetBpm', target_bpm, ...
        'targetAngleDegree', target_angle, 'angleThresholdDegree', angleThresholdDegree, ...
        'filter_range', fr, 'secondDerivative', secondDerivativeFlag);
    candidateHistory = candidateHistory(candidateHistory(:, 5) ~= 0, :);
end

% Select the spatial midpoint of all candidate bins
history_list = candidateHistory;
range_s = min(history_list(:, 3));
range_l = max(history_list(:, 3));
angle_s = min(history_list(:, 4));
angle_l = max(history_list(:, 4));
range_m = floor(range_s + (range_l - range_s) / 2);
angle_m = floor(angle_s + (angle_l - angle_s) / 2);

found_bin = [range_m, angle_m];

% Crop and save head data
head_angleBin_threshold = 10;
head_rangeBin_threshold = 2;
data = imgs(cut_index(1, 1):cut_index(1, 2), ...
    range_m - head_rangeBin_threshold:range_m + head_rangeBin_threshold, ...
    angle_m - head_angleBin_threshold:angle_m + head_angleBin_threshold);
save(fullfile(SAVE_DIR, 'head_example'), 'data');

%% ========================================================================
%  Neck -- magnitude-ranked bin search
%  ========================================================================
neck_distance        = 0.60;       % expected distance to neck (m)
target_angle         = -20;
angleThresholdDegree = 20;
fr                   = [0.5, 10];
secondDerivativeFlag = 1;

[candidateHistory, ~, fs] = find_bin(imgs, neck_distance, ...
    cut_index(1, 1), cut_index(1, 2), ...
    'thresholdOnBpm', thresholdOnBpm, 'targetBpm', target_bpm, ...
    'secondDerivative', secondDerivativeFlag, 'filter_range', fr, ...
    'targetAngleDegree', target_angle, 'angleThresholdDegree', angleThresholdDegree);

% Keep candidates with non-zero lag, rank by signal magnitude
candidateHistory = candidateHistory(candidateHistory(:, 5) ~= 0, :);
candidateHistory = sortrows(candidateHistory, 2, 'descend');
N = 40;
history_list = candidateHistory(1:min(N, size(candidateHistory, 1)), :);

found_bin = [history_list(1, 3), history_list(1, 4)];

% Crop radar data around the selected neck bin
neck_bin_threshold = 10;
best_range = history_list(1, 3);
best_angle = history_list(1, 4);
if min(128, best_angle + neck_bin_threshold) == 128
    data = imgs(cut_index(1, 1):cut_index(1, 2), best_range, end-neck_bin_threshold*2:end);
else
    data = imgs(cut_index(1, 1):cut_index(1, 2), best_range, ...
        best_angle - neck_bin_threshold:min(128, best_angle + neck_bin_threshold));
end
save(fullfile(SAVE_DIR, 'neck_example'), 'data');
