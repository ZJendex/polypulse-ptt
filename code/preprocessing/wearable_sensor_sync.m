% wearable_sensor_sync.m  Stage 1 -- Synchronise wearable sensors and extract cardiac peaks.
%
% This script:
%   1. Loads raw wearable data (4 IMU + PPG channels) and cleans timestamps.
%   2. Aligns sensors 2-4 to sensor 1 via sliding-window cross-correlation
%      during a calibration interval.
%   3. Detects PPG foot points (intersection-tangent method), then locates
%      corresponding SCG, BCG, and neck-pulse peaks for each heartbeat.
%   4. Refines peak locations using histogram-based PTT narrowing.
%   5. Removes amplitude-based outliers from each peak set.
%
% Workspace variables produced for later stages:
%   cali_time, end_cutting_time, fs  -- timing constants
%   imu1..4, t_i1..4                 -- cleaned IMU signals and timestamps
%   o1..4                            -- outlier indices from raw data
%   post_sensor_cali_time            -- end of last sensor calibration event
%   time_axis_scg/ppg/bcg/neck       -- per-modality time vectors
%   scg_data, ppg_data, bcg_data, neck_data
%   peak_index_scg/ppg/bcg/neck      -- sample indices of detected peaks
%   peak_time_scg/ppg/bcg/neck       -- peak timestamps (post-cali only)
%   peak_time_scg_op/ppg_op/bcg_op/neck_op  -- full peak timestamps (for radar sync)

%% ========================================================================
%  Configuration
%  ========================================================================
cali_time        = 90;   % calibration duration (s)
end_cutting_time = 170;  % end of analysis window (s)
fs               = 500;  % wearable sampling rate (Hz)
SAMPLE_PERIOD    = 1/fs; % 0.002 s

%% ========================================================================
%  1. Load and clean raw wearable data
%  ========================================================================
load(fullfile('..', '..', 'data', 'example', 'original_sensor_data', 'wearable_data.mat'))

imu1 = detrend(y);
imu2 = detrend(y2);
imu3 = detrend(y3);
imu4 = detrend(y4);
t_i1 = double(time_stamp)  / 1000;
t_i2 = double(time_stamp2) / 1000;
t_i3 = double(time_stamp3) / 1000;
t_i4 = double(time_stamp4) / 1000;

% Trim first 4 and last sample (transient / boundary artefacts)
imu1 = imu1(5:end-1);  imu2 = imu2(5:end-1);
imu3 = imu3(5:end-1);  imu4 = imu4(5:end-1);
t_i1 = t_i1(5:end-1);  t_i2 = t_i2(5:end-1);
t_i3 = t_i3(5:end-1);  t_i4 = t_i4(5:end-1);

% Zero-base each timestamp vector
t_i1 = t_i1 - t_i1(1);
t_i2 = t_i2 - t_i2(1);
t_i3 = t_i3 - t_i3(1);
t_i4 = t_i4 - t_i4(1);

% Remove gross outlier samples
[t_i1, imu1, o1] = outlierRemovalForDataPairs(t_i1, imu1);
[t_i2, imu2, o2] = outlierRemovalForDataPairs(t_i2, imu2);
[t_i3, imu3, o3] = outlierRemovalForDataPairs(t_i3, imu3);
[t_i4, imu4, o4] = outlierRemovalForDataPairs(t_i4, imu4);

% Repair early-sample timestamp glitches (sensors sometimes report bogus
% timestamps for the first few samples before the clock stabilises)
t_i1 = repairEarlyTimestamps(t_i1, SAMPLE_PERIOD);
t_i2 = repairEarlyTimestamps(t_i2, SAMPLE_PERIOD);
t_i3 = repairEarlyTimestamps(t_i3, SAMPLE_PERIOD);
t_i4 = repairEarlyTimestamps(t_i4, SAMPLE_PERIOD);

% Re-zero after repair
t_i1 = t_i1 - t_i1(1);
t_i2 = t_i2 - t_i2(1);
t_i3 = t_i3 - t_i3(1);
t_i4 = t_i4 - t_i4(1);

figure(111)
plot(t_i1, imu1); hold on
plot(t_i2, imu2);
plot(t_i3, imu3);
plot(t_i4, imu4);
legend('Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4');
hold off
title('Raw IMU signals after cleaning');

%% ========================================================================
%  2. Cross-correlate IMU pairs to find per-sensor time shifts
%  ========================================================================
% During calibration, each sensor is tapped in sequence.  A sliding window
% cross-correlation finds the tap event that yields the strongest peak
% between sensor 1 and each other sensor.
step  = 0.2;   % sliding-window step (s)
w_len = 1.5;   % sliding-window length (s)
numWindows = floor(cali_time / step) + 1;

p_shift_bins12 = zeros(numWindows, 3);
p_shift_bins13 = zeros(numWindows, 3);
p_shift_bins14 = zeros(numWindows, 3);
k = 1;

for w_start = 0:step:cali_time
    xl = [w_start, w_start + w_len];

    cut_imu1 = imu1(t_i1 > xl(1) & t_i1 < xl(2));
    cut_t_i1 = t_i1(t_i1 > xl(1) & t_i1 < xl(2));
    cut_imu2 = imu2(t_i2 > xl(1) & t_i2 < xl(2));
    cut_t_i2 = t_i2(t_i2 > xl(1) & t_i2 < xl(2));
    cut_imu3 = imu3(t_i3 > xl(1) & t_i3 < xl(2));
    cut_t_i3 = t_i3(t_i3 > xl(1) & t_i3 < xl(2));
    cut_imu4 = imu4(t_i4 > xl(1) & t_i4 < xl(2));
    cut_t_i4 = t_i4(t_i4 > xl(1) & t_i4 < xl(2));

    % Smooth any remaining timestamp jumps within the window
    cut_t_i1 = smoothTimestamps(cut_t_i1, SAMPLE_PERIOD);

    % Resample all channels onto a uniform grid and detrend
    [cut_n_imu_i1, ~] = interpolationWithTimestamp(cut_imu1, cut_t_i1, fs);
    [cut_n_imu_i2, ~] = interpolationWithTimestamp(cut_imu2, cut_t_i2, fs);
    [cut_n_imu_i3, ~] = interpolationWithTimestamp(cut_imu3, cut_t_i3, fs);
    [cut_n_imu_i4, ~] = interpolationWithTimestamp(cut_imu4, cut_t_i4, fs);
    cut_n_imu_i1 = detrend(cut_n_imu_i1);
    cut_n_imu_i2 = detrend(cut_n_imu_i2);
    cut_n_imu_i3 = detrend(cut_n_imu_i3);
    cut_n_imu_i4 = detrend(cut_n_imu_i4);

    % Truncate to common length
    s_len = min([length(cut_n_imu_i1), length(cut_n_imu_i2), ...
                 length(cut_n_imu_i3), length(cut_n_imu_i4)]);
    cut_n_imu_i1 = cut_n_imu_i1(1:s_len);
    cut_n_imu_i2 = cut_n_imu_i2(1:s_len);
    cut_n_imu_i3 = cut_n_imu_i3(1:s_len);
    cut_n_imu_i4 = cut_n_imu_i4(1:s_len);

    % Cross-correlate sensor 1 with sensors 2, 3, 4
    p_shift_bins12(k, :) = xcorrShiftForWindow(cut_n_imu_i1, cut_n_imu_i2, w_start);
    p_shift_bins13(k, :) = xcorrShiftForWindow(cut_n_imu_i1, cut_n_imu_i3, w_start);
    p_shift_bins14(k, :) = xcorrShiftForWindow(cut_n_imu_i1, cut_n_imu_i4, w_start);
    k = k + 1;
end

% Select the window with the strongest correlation for each pair
[~, shift12_index] = max(p_shift_bins12(:, 2));
shift_bins12       = p_shift_bins12(shift12_index, 1);
shift_bins12_start = p_shift_bins12(shift12_index, 3);

[~, shift13_index] = max(p_shift_bins13(:, 2));
shift_bins13       = p_shift_bins13(shift13_index, 1);
shift_bins13_start = p_shift_bins13(shift13_index, 3);

[~, shift14_index] = max(p_shift_bins14(:, 2));
shift_bins14       = p_shift_bins14(shift14_index, 1);
shift_bins14_start = p_shift_bins14(shift14_index, 3);

%% Resolve calibration collisions
% When two calibration events land within 1 second of each other, the
% algorithm keeps the one with higher correlation and re-searches the
% weaker one at least step_away seconds from the stronger one.
step_away = 1;

% --- Pair 1-2 vs Pair 1-3 ---
if abs(shift_bins13_start - shift_bins12_start) < 1
    if p_shift_bins12(shift12_index, 2) > p_shift_bins13(shift13_index, 2)
        [~, shift13_index] = max(p_shift_bins13(:, 2));
        while abs(p_shift_bins13(shift13_index, 3) - shift_bins12_start) < step_away
            p_shift_bins13(shift13_index, 2) = 0;
            [~, shift13_index] = max(p_shift_bins13(:, 2));
        end
        shift_bins13       = p_shift_bins13(shift13_index, 1);
        shift_bins13_start = p_shift_bins13(shift13_index, 3);
    end
    if p_shift_bins12(shift12_index, 2) < p_shift_bins13(shift13_index, 2)
        [~, shift12_index] = max(p_shift_bins12(:, 2));
        while abs(p_shift_bins12(shift12_index, 3) - shift_bins13_start) < step_away
            p_shift_bins12(shift12_index, 2) = 0;
            [~, shift12_index] = max(p_shift_bins12(:, 2));
        end
        shift_bins12       = p_shift_bins12(shift12_index, 1);
        shift_bins12_start = p_shift_bins12(shift12_index, 3);
    end
end

% --- Pair 1-3 vs Pair 1-4 ---
if abs(shift_bins13_start - shift_bins14_start) < 1
    if p_shift_bins14(shift14_index, 2) > p_shift_bins13(shift13_index, 2)
        [~, shift13_index] = max(p_shift_bins13(:, 2));
        while abs(p_shift_bins13(shift13_index, 3) - shift_bins14_start) < step_away
            p_shift_bins13(shift13_index, 2) = 0;
            [~, shift13_index] = max(p_shift_bins13(:, 2));
        end
        shift_bins13       = p_shift_bins13(shift13_index, 1);
        shift_bins13_start = p_shift_bins13(shift13_index, 3);
    end
    if p_shift_bins14(shift14_index, 2) < p_shift_bins13(shift13_index, 2)
        [~, shift14_index] = max(p_shift_bins14(:, 2));
        while abs(p_shift_bins14(shift14_index, 3) - shift_bins13_start) < step_away
            p_shift_bins14(shift14_index, 2) = 0;
            [~, shift14_index] = max(p_shift_bins14(:, 2));
        end
        shift_bins14       = p_shift_bins14(shift14_index, 1);
        shift_bins14_start = p_shift_bins14(shift14_index, 3);
    end
end

% --- Pair 1-2 vs Pair 1-4 ---
if abs(shift_bins12_start - shift_bins14_start) < 1
    if p_shift_bins14(shift14_index, 2) > p_shift_bins12(shift12_index, 2)
        [~, shift13_index] = max(p_shift_bins13(:, 2));  %#ok  (inherited from original code)
        while abs(p_shift_bins12(shift12_index, 3) - shift_bins14_start) < step_away
            p_shift_bins12(shift12_index, 2) = 0;
            [~, shift12_index] = max(p_shift_bins12(:, 2));
        end
        shift_bins12       = p_shift_bins12(shift12_index, 1);
        shift_bins12_start = p_shift_bins12(shift12_index, 3);
    end
    if p_shift_bins14(shift14_index, 2) < p_shift_bins12(shift12_index, 2)
        [~, shift14_index] = max(p_shift_bins14(:, 2));
        while abs(p_shift_bins14(shift14_index, 3) - shift_bins12_start) < step_away
            p_shift_bins14(shift14_index, 2) = 0;
            [~, shift14_index] = max(p_shift_bins14(:, 2));
        end
        shift_bins14       = p_shift_bins14(shift14_index, 1);
        shift_bins14_start = p_shift_bins14(shift14_index, 3);
    end
end

% Apply time shifts to align sensors 2-4 to sensor 1
t_i2 = t_i2 + shift_bins12 / fs;
t_i3 = t_i3 + shift_bins13 / fs;
t_i4 = t_i4 + shift_bins14 / fs;

%% Verification plot: overlay aligned sensor pairs at calibration windows
figure(335)
subplot(3,1,1)
plotAlignedPair(imu1, t_i1, imu2, t_i2, shift_bins12_start, w_len, fs, 'Sensor 1 & Sensor 2');
subplot(3,1,2)
plotAlignedPair(imu1, t_i1, imu3, t_i3, shift_bins13_start, w_len, fs, 'Sensor 1 & Sensor 3');
subplot(3,1,3)
plotAlignedPair(imu1, t_i1, imu4, t_i4, shift_bins14_start, w_len, fs, 'Sensor 1 & Sensor 4');

%% ========================================================================
%  3. Prepare ground-truth signal channels and extract cardiac peaks
%  ========================================================================
post_sensor_cali_time = floor(max([shift_bins12_start, shift_bins13_start, shift_bins14_start]));

time_axis_scg  = t_i1;
time_axis_ppg  = t_i2;
time_axis_bcg  = t_i3;
time_axis_neck = t_i4;

time_duration_gt = 180;

% Extract and clean each physiological signal channel
scg_data  = z(5:end-1);   scg_data(o1) = [];
ppg_data  = double(ADC_values2); ppg_data = ppg_data(5:end-1); ppg_data(o2) = [];
bcg_data  = x3(5:end-1);  bcg_data(o3) = [];
neck_data = z4(5:end-1);  neck_data(o4) = [];

[time_axis_scg,  scg_data,  ~] = outlierRemovalForDataPairs(time_axis_scg,  scg_data);
[time_axis_ppg,  ppg_data,  ~] = outlierRemovalForDataPairs(time_axis_ppg,  ppg_data);
[time_axis_bcg,  bcg_data,  ~] = outlierRemovalForDataPairs(time_axis_bcg,  bcg_data);
[time_axis_neck, neck_data, ~] = outlierRemovalForDataPairs(time_axis_neck, neck_data);

% Per-channel effective sampling rates
fs_scg  = length(time_stamp)  / time_duration_gt;
fs_ppg  = length(time_stamp2) / time_duration_gt;
fs_bcg  = length(time_stamp3) / time_duration_gt;
fs_neck = length(time_stamp4) / time_duration_gt;

%% ---- 3a. PPG peak detection (intersection-tangent method) ---------------
fc = 10;
[b, a] = butter(4, fc/(fs_ppg/2), 'low');
smoothed_ppg = filtfilt(b, a, ppg_data);
first_derivative = computeFirstDerivative(smoothed_ppg, 1/fs_ppg);

[~, peak_index_ppg, ~, prom] = findpeaks(normalize(first_derivative, 'range'), ...
    'MinPeakDistance', 0.4*fs_ppg, 'MinPeakProminence', 0.001);

% Refine detection thresholds from initial-pass statistics
HBV_threshold = 0.2;  % inter-beat interval tolerance (s)
refined_MPD = mean(diff(peak_index_ppg))/fs - HBV_threshold;
refined_MPP = median(prom) * 0.5;
[~, peak_index_ppg] = findpeaks(normalize(first_derivative, 'range'), ...
    'MinPeakDistance', refined_MPD*fs_ppg, 'MinPeakProminence', refined_MPP);

% Keep only peaks after the calibration interval
peak_index_ppg = peak_index_ppg(peak_index_ppg > cali_time * fs_ppg);

% Locate valleys between successive derivative peaks
valley_index_ppg = zeros(1, length(peak_index_ppg)-1);
for i = 1:length(peak_index_ppg)-1
    searchRange = peak_index_ppg(i):peak_index_ppg(i+1);
    [~, minIdx] = min(smoothed_ppg(searchRange));
    valley_index_ppg(i) = searchRange(minIdx);
end

% Find PPG foot as intersection of the peak tangent with the valley level
intersection_points = zeros(1, length(valley_index_ppg));
for i = 1:length(valley_index_ppg)
    peakIdx   = peak_index_ppg(i+1);
    valleyIdx = valley_index_ppg(i);
    tangentSlope     = (smoothed_ppg(peakIdx+1) - smoothed_ppg(peakIdx-1)) / 2;
    tangentIntercept = smoothed_ppg(peakIdx) - tangentSlope * peakIdx;
    intersection_points(i) = round((smoothed_ppg(valleyIdx) - tangentIntercept) / tangentSlope);
end
peak_index_ppg = intersection_points;
peak_index_ppg = peak_index_ppg(~isnan(peak_index_ppg));

% Remove PPG inter-beat outliers (physiological IBI range ~ 300-1200 ms)
ppg_hbi = diff(peak_index_ppg);
outlier_mask = ppg_hbi < median(ppg_hbi) - 0.3*fs_ppg | ppg_hbi > median(ppg_hbi) + 0.3*fs_ppg;
outlier_mask = logical([outlier_mask, 0]);
peak_index_ppg(outlier_mask) = [];

%% ---- 3b. SCG peak detection (aligned to PPG) ---------------------------
fc = 0.7;
[b, a] = butter(1, fc/(fs_scg/2), 'high');
scg_data = filtfilt(b, a, scg_data);
fc = 30;
[b, a] = butter(1, fc/(fs_scg/2), 'low');
scg_data = filtfilt(b, a, scg_data);

% Sweep candidate PPG-to-SCG delays and pick the delay that maximises
% the sum of SCG amplitudes at detected peaks.
refPPG_delay_range = [0.1, 0.3];
step_refPPG        = 0.02;
[peak_index_scg, ~] = findPeaksRelativeToPPG(scg_data, time_axis_scg, ...
    peak_index_ppg, time_axis_ppg, refPPG_delay_range, step_refPPG, 0, 0.05);

%% ---- 3c. BCG peak detection (aligned to PPG) ---------------------------
fc = 0.5;
[b, a] = butter(1, fc/(fs_bcg/2), 'high');
bcg_data = filtfilt(b, a, bcg_data);
fc = 50;
[b, a] = butter(1, fc/(fs_bcg/2), 'low');
bcg_data = filtfilt(b, a, bcg_data);
bcg_data = -bcg_data;  % invert to match expected polarity

[peak_index_bcg, ~] = findPeaksRelativeToPPG(bcg_data, time_axis_bcg, ...
    peak_index_ppg, time_axis_ppg, refPPG_delay_range, step_refPPG, 0.1, 0.05);

%% ---- 3d. Neck pulse peak detection (aligned to PPG) --------------------
fc = 0.7;
[b, a] = butter(1, fc/(fs_neck/2), 'high');
neck_data = filtfilt(b, a, neck_data);
fc = 30;
[b, a] = butter(1, fc/(fs_neck/2), 'low');
neck_data = filtfilt(b, a, neck_data);

[peak_index_neck, ~] = findPeaksRelativeToPPG(neck_data, time_axis_neck, ...
    peak_index_ppg, time_axis_ppg, refPPG_delay_range, step_refPPG, 0.1, 0.05);

%% ========================================================================
%  4. Compute rough PTT and refine peaks via histogram narrowing
%  ========================================================================
peak_time_scg  = time_axis_scg(peak_index_scg);
peak_time_scg  = peak_time_scg(peak_time_scg > cali_time);
peak_time_ppg  = time_axis_ppg(peak_index_ppg);
peak_time_ppg  = peak_time_ppg(peak_time_ppg > cali_time);
peak_time_bcg  = time_axis_bcg(peak_index_bcg);
peak_time_bcg  = peak_time_bcg(peak_time_bcg > cali_time);
peak_time_neck = time_axis_neck(peak_index_neck);
peak_time_neck = peak_time_neck(peak_time_neck > cali_time);

% Trim to common time range
ppgOverRange = 0.2;
peak_time_scg  = peak_time_scg(peak_time_scg   < max(peak_time_ppg) + ppgOverRange);
peak_time_bcg  = peak_time_bcg(peak_time_bcg   < max(peak_time_ppg) + ppgOverRange);
peak_time_neck = peak_time_neck(peak_time_neck < max(peak_time_ppg) + ppgOverRange);
n_peaks = min([length(peak_time_scg), length(peak_time_ppg), ...
               length(peak_time_bcg), length(peak_time_neck)]);
peak_time_scg  = peak_time_scg(end-n_peaks+1:end);
peak_time_ppg  = peak_time_ppg(end-n_peaks+1:end);
peak_time_bcg  = peak_time_bcg(end-n_peaks+1:end);
peak_time_neck = peak_time_neck(end-n_peaks+1:end);

gt_ptt1 = peak_time_ppg - peak_time_scg;   % Heart -> Wrist
gt_ptt2 = peak_time_bcg - peak_time_scg;   % Heart -> Head
gt_ptt3 = peak_time_ppg - peak_time_bcg;   % Head  -> Wrist
gt_ptt4 = peak_time_ppg - peak_time_neck;  % Neck  -> Wrist

% Refine SCG peaks using the histogram mode of PTT(PPG-SCG)
peak_index_scg = refinePeaksViaHistogram(scg_data, time_axis_scg, ...
    peak_index_ppg, time_axis_ppg, gt_ptt1, 0.03, 0.05);

% Refine BCG peaks using the histogram mode of PTT(PPG-BCG)
peak_index_bcg = refinePeaksViaHistogram(bcg_data, time_axis_bcg, ...
    peak_index_ppg, time_axis_ppg, gt_ptt3, 0.05, 0.05);

% Refine neck peaks using the histogram mode of PTT(PPG-Neck)
peak_index_neck = refinePeaksViaHistogram(neck_data, time_axis_neck, ...
    peak_index_ppg, time_axis_ppg, gt_ptt4, 0.04, 0.05);

%% ========================================================================
%  5. Final PTT computation and outlier removal
%  ========================================================================
peak_time_scg     = time_axis_scg(peak_index_scg);
peak_time_scg_op  = peak_time_scg;
peak_time_scg     = peak_time_scg(peak_time_scg > cali_time);

peak_time_ppg     = time_axis_ppg(peak_index_ppg);
peak_time_ppg_op  = peak_time_ppg;
peak_time_ppg     = peak_time_ppg(peak_time_ppg > cali_time);

peak_time_bcg     = time_axis_bcg(peak_index_bcg);
peak_time_bcg_op  = peak_time_bcg;
peak_time_bcg     = peak_time_bcg(peak_time_bcg > cali_time);

peak_time_neck    = time_axis_neck(peak_index_neck);
peak_time_neck_op = peak_time_neck;
peak_time_neck    = peak_time_neck(peak_time_neck > cali_time);

% Trim to common time range
peak_time_scg  = peak_time_scg(peak_time_scg   < max(peak_time_ppg) + ppgOverRange);
peak_time_bcg  = peak_time_bcg(peak_time_bcg   < max(peak_time_ppg) + ppgOverRange);
peak_time_neck = peak_time_neck(peak_time_neck < max(peak_time_ppg) + ppgOverRange);
n_peaks = min([length(peak_time_scg), length(peak_time_ppg), ...
               length(peak_time_bcg), length(peak_time_neck)]);
peak_time_scg  = peak_time_scg(end-n_peaks+1:end);
peak_time_ppg  = peak_time_ppg(end-n_peaks+1:end);
peak_time_bcg  = peak_time_bcg(end-n_peaks+1:end);
peak_time_neck = peak_time_neck(end-n_peaks+1:end);

gt_ptt1 = peak_time_ppg - peak_time_scg;
gt_ptt2 = peak_time_bcg - peak_time_scg;
gt_ptt3 = peak_time_ppg - peak_time_bcg;
gt_ptt4 = peak_time_ppg - peak_time_neck;
gt_ptt5 = peak_time_neck - peak_time_scg;
gt_ptt6 = peak_time_bcg - peak_time_neck;

median_gt_heart2Wrist_ptt = median(gt_ptt1);
median_gt_heart2Head_ptt  = median(gt_ptt2);
median_gt_head2Wrist_ptt  = median(gt_ptt3);
median_gt_neck2Wrist_ptt  = median(gt_ptt4);
median_gt_heart2Neck_ptt  = median(gt_ptt5);

% Remove peaks whose high-pass amplitude exceeds numStd standard deviations
% from the median (per-channel 0.6 Hz high-pass isolates cardiac component)
numStd = 2;
peak_index_scg  = removeAmplitudeOutliers(scg_data,  peak_index_scg,  fs_scg,  numStd, 0.6);
peak_index_ppg  = removeAmplitudeOutliers(ppg_data,  peak_index_ppg,  fs_ppg,  numStd, 0.6);
peak_index_bcg  = removeAmplitudeOutliers(bcg_data,  peak_index_bcg,  fs_bcg,  numStd, 0.6);
peak_index_neck = removeAmplitudeOutliers(neck_data, peak_index_neck, fs_neck, numStd, 0.6);


%% ========================================================================
%  Local helper functions  (MATLAB R2016b+ allows local functions in scripts)
%  ========================================================================

function t = repairEarlyTimestamps(t, samplePeriod)
% repairEarlyTimestamps  Back-fill bogus early timestamps.
%   Some sensors report near-zero timestamps for the first few samples
%   before the clock stabilises.  This detects the issue by checking
%   whether the 100th sample exceeds 1 s, then back-fills using the
%   expected sample period.
    CHECK_LIMIT = 100;
    if t(CHECK_LIMIT) > 1
        validIdx = find(t > 1, 1, 'first');
        if ~isempty(validIdx) && validIdx < CHECK_LIMIT
            startVal = t(validIdx);
            t(1:validIdx-1) = startVal - samplePeriod * ((validIdx-1):-1:1);
        end
    end
end

function t = smoothTimestamps(t, samplePeriod)
% smoothTimestamps  Fix non-monotonic or excessively large timestamp gaps.
    for idx = 2:length(t)
        gap = t(idx) - t(idx-1);
        if gap < 0 || gap > 0.01
            t(idx) = t(idx-1) + samplePeriod;
        end
    end
end

function row = xcorrShiftForWindow(sigA, sigB, windowStart)
% xcorrShiftForWindow  Return [lag, maxAcf, windowStart] for one window.
    [acf, lags] = xcorr(sigA, sigB);
    [vals, idxs] = findpeaks(acf);
    if ~isempty(vals)
        [~, bestIdx] = max(vals);
        row = [lags(idxs(bestIdx)), max(acf), windowStart];
    else
        row = [0, -inf, windowStart];
    end
end

function plotAlignedPair(imu1, t1, imu2, t2, winStart, winLen, fs, titleStr)
% plotAlignedPair  Overlay two normalised IMU signals for visual check.
    xl = [winStart, winStart + winLen];
    c1  = imu1(t1 > xl(1) & t1 < xl(2));
    ct1 = t1(t1 > xl(1) & t1 < xl(2));
    c2  = imu2(t2 > xl(1) & t2 < xl(2));
    ct2 = t2(t2 > xl(1) & t2 < xl(2));
    [n1, nt1] = interpolationWithTimestamp(c1, ct1, fs);
    [n2, nt2] = interpolationWithTimestamp(c2, ct2, fs);
    n1 = detrend(normalize(n1, 'range'));
    n2 = detrend(normalize(n2, 'range'));
    plot(nt1, n1); hold on; plot(nt2, n2); hold off;
    title(titleStr);
end

function [peakIndices, bestDelay] = findPeaksRelativeToPPG(signalData, timeAxis, ...
        ppgPeakIdx, ppgTimeAxis, delayRange, delayStep, extraOffset, minProm)
% findPeaksRelativeToPPG  Detect peaks in signalData near each PPG peak.
%   Sweeps over candidate delays from the PPG peak, picks the delay that
%   maximises the total signal amplitude at detected peaks, then runs a
%   final pass at the best delay.
%
%   extraOffset: additional time (s) to extend the search window past the
%                PPG peak (0 for SCG, 0.1 for BCG/neck).
    numDelaySteps = floor(diff(delayRange) / delayStep);
    amplitudeSum  = zeros(numDelaySteps, 1);
    peakIndices   = zeros(1, length(ppgPeakIdx));

    k = 1;
    for delay = delayRange(1):delayStep:delayRange(2)
        for i = 1:length(ppgPeakIdx)
            ppgTime   = ppgTimeAxis(ppgPeakIdx(i));
            searchIdx = find((ppgTime - delay < timeAxis) & ...
                             (timeAxis < ppgTime + extraOffset));
            if isempty(searchIdx) || length(searchIdx) < 3
                last_nz = find(peakIndices ~= 0, 1, 'last');
                if ~isempty(last_nz)
                    peakIndices = peakIndices(1:last_nz);
                end
                break
            end
            [~, pkSec] = findpeaks(signalData(searchIdx), ...
                'MinPeakDistance', length(searchIdx) - 2, ...
                'MinPeakProminence', minProm);
            if isempty(pkSec)
                pkSec = 0;
            end
            peakIndices(i) = searchIdx(1) + pkSec;
        end
        amplitudeSum(k) = sum(signalData(peakIndices));
        k = k + 1;
    end

    [~, bestIdx] = max(amplitudeSum);
    bestDelay = delayRange(1) + delayStep * bestIdx;

    % Final pass at the best delay with max-fallback when findpeaks fails
    for i = 1:length(ppgPeakIdx)
        ppgTime   = ppgTimeAxis(ppgPeakIdx(i));
        searchIdx = find((ppgTime - bestDelay < timeAxis) & ...
                         (timeAxis < ppgTime + extraOffset));
        if isempty(searchIdx)
            last_nz = find(peakIndices ~= 0, 1, 'last');
            if ~isempty(last_nz)
                peakIndices = peakIndices(1:last_nz);
            end
            break
        end
        [~, pkSec] = findpeaks(signalData(searchIdx), ...
            'MinPeakDistance', length(searchIdx) - 2, ...
            'MinPeakProminence', minProm);
        if isempty(pkSec)
            [~, maxIdx] = max(signalData(searchIdx));
            pkSec = maxIdx;
        end
        peakIndices(i) = searchIdx(1) + pkSec;
    end
end

function peakIdx = refinePeaksViaHistogram(signalData, timeAxis, ...
        ppgPeakIdx, ppgTimeAxis, pttValues, searchWidth, minProm)
% refinePeaksViaHistogram  Narrow the peak search window using PTT histogram mode.
%   Bins the rough PTT values, finds the modal bin, and re-searches peaks
%   within a tight window centred on the modal delay.
    binWidth = 0.01;
    [binCounts, binEdges] = histcounts(pttValues, 'BinWidth', binWidth);
    [~, maxBinIdx] = max(binCounts);
    modeDelay = (binEdges(maxBinIdx) + binEdges(maxBinIdx + 1)) / 2;

    peakIdx = zeros(1, length(ppgPeakIdx));
    for i = 1:length(ppgPeakIdx)
        ppgTime   = ppgTimeAxis(ppgPeakIdx(i));
        searchIdx = find((ppgTime - modeDelay - searchWidth < timeAxis) & ...
                         (timeAxis < ppgTime - modeDelay + searchWidth));
        if isempty(searchIdx)
            last_nz = find(peakIdx ~= 0, 1, 'last');
            if ~isempty(last_nz)
                peakIdx = peakIdx(1:last_nz);
            end
            break
        end
        [~, pkSec] = findpeaks(signalData(searchIdx), ...
            'MinPeakDistance', length(searchIdx) - 2, ...
            'MinPeakProminence', minProm);
        if isempty(pkSec)
            [~, maxIdx] = max(signalData(searchIdx));
            pkSec = maxIdx;
        end
        peakIdx(i) = searchIdx(1) + pkSec;
    end
end

function peakIdx = removeAmplitudeOutliers(data, peakIdx, fs, numStd, fc)
% removeAmplitudeOutliers  Remove peaks whose high-pass amplitude is an outlier.
    [b, a] = butter(4, fc/(fs/2), 'high');
    filteredData = filtfilt(b, a, data);
    peakVals  = filteredData(peakIdx);
    isOutlier = peakVals > median(peakVals) + numStd*std(peakVals) | ...
                peakVals < median(peakVals) - numStd*std(peakVals);
    isOutlier = logical([isOutlier, 0]);
    peakIdx(isOutlier) = [];
end
