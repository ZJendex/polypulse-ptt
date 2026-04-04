% radar_sensor_sync.m  Stage 2 -- Synchronise radar clock with wearable sensors.
%
% Loads the radar data cube, then estimates the time offset between the
% radar and wearable sensor clocks by correlating the second derivative of
% radar displacement with IMU acceleration during the calibration window.
% The estimated shift is applied to all wearable peak timestamps and
% time axes so that subsequent stages operate in the radar time base.
%
% Requires workspace variables from Stage 1:
%   z2, o2, t_i2              -- raw z-axis signal, outlier indices, timestamps
%   cali_time, fs              -- calibration duration (s) and wearable sampling rate (Hz)
%   post_sensor_cali_time      -- end of last sensor calibration event (s)
%   time_axis_scg/ppg/bcg/neck -- per-modality time vectors
%   peak_time_scg_op/ppg_op/bcg_op/neck_op -- full peak timestamps
%
% Produces:
%   imgs                                   -- 3-D radar data cube
%   time_axis_scg_s/ppg_s/bcg_s/neck_s     -- shifted time axes
%   peak_time_scg/ppg/bcg/neck             -- shifted peak timestamps

%% Load radar data and configuration
load(fullfile('..', '..', 'data', 'example', 'original_sensor_data', 'radar_data.mat'))
cfg = getRadarConfig(imgs);

% Prepare IMU reference signal from sensor 2 (collocated with radar during calibration)
radar_cali = z2(5:end-1);
radar_cali(o2) = [];
imu  = detrend(radar_cali);
t_i  = t_i2;
hard_code_radar_range = 7;

%% Find best radar-to-IMU time shift via sliding-window cross-correlation
% Search over angle bins near the expected radar boresight to find the
% angle that yields the highest correlation with the IMU.
sync_backward_time = 3;  % extend search before post_sensor_cali_time (s)
xl_r = [post_sensor_cali_time - sync_backward_time, cali_time];

angle_bin_range = [50, 60];
angles_shift_bins = zeros(diff(angle_bin_range), 3);
angle_index = 1;

for angle_bin = angle_bin_range(1):angle_bin_range(2)
    % Extract and process radar displacement at this angle bin
    [dist, t] = plot_waveformOnBin([hard_code_radar_range, angle_bin], imgs, ...
        xl_r(1)*cfg.fs, xl_r(2)*cfg.fs, ...
        'flipped', 0, 'filter', 1, 'filter_range', [0.7, 30], 'phase', 1, 'plot', 0);
    d_dist = computeSecondDerivative(dist, 1/cfg.fs);

    % Low-pass filter to isolate vertical gripping motion
    fc = 10;
    [b, a] = butter(4, fc/(cfg.fs/2), 'low');
    d_dist = filtfilt(b, a, d_dist);

    % Sliding-window cross-correlation between radar acceleration and IMU
    step  = 0.5;
    w_len = 3;
    p_shift_bins = zeros(floor((xl_r(2) - xl_r(1) - step) / step), 3);
    k = 1;

    for w_start = 0:step:xl_r(2) - xl_r(1) - step
        xl = [w_start, w_start + w_len];

        cut_d_dist = d_dist(t > xl(1) & t < xl(2));
        cut_imu    = imu(t_i > xl(1) + xl_r(1) & t_i < xl(2) + xl_r(1));
        cut_t_i    = t_i(t_i > xl(1) + xl_r(1) & t_i < xl(2) + xl_r(1));

        % Resample IMU onto a uniform grid
        % (fs == cfg.fs == 500 Hz: wearable and radar rates match by design)
        [cut_n_imu_i, ~] = interpolationWithTimestamp(cut_imu, cut_t_i, fs);
        cut_n_d_dist = detrend(normalize(cut_d_dist, 'range'));
        cut_n_imu    = detrend(normalize(cut_n_imu_i, 'range'));

        s_len = min(length(cut_n_d_dist), length(cut_n_imu));
        cut_n_d_dist = cut_n_d_dist(1:s_len);
        cut_n_imu    = cut_n_imu(1:s_len);

        [acf, lags] = xcorr(cut_n_d_dist, cut_n_imu);
        [val, index] = findpeaks(acf);
        if ~isempty(val)
            [~, im] = max(val);
            p_shift_bins(k, :) = [lags(index(im)), max(acf), w_start];
        else
            p_shift_bins(k, 2) = -inf;
        end
        k = k + 1;
    end

    % Select the most confident window using a 3-window moving average of
    % correlation strength (ensures the shift comes from the middle of the
    % calibration event, not its edge)
    NMSEs = p_shift_bins(:, 2);
    sliding_sum = NMSEs(1:end-2) + NMSEs(2:end-1) + NMSEs(3:end);
    [~, shift_index] = max(sliding_sum);
    shift_index = shift_index + 1;  % offset for the 3-element window

    angles_shift_bins(angle_index, 1) = p_shift_bins(shift_index, 1);
    angles_shift_bins(angle_index, 2) = p_shift_bins(shift_index, 2);
    angles_shift_bins(angle_index, 3) = p_shift_bins(shift_index, 3);
    angle_index = angle_index + 1;
end

%% Select the angle bin with the strongest overall correlation
[~, selected_angle_index] = max(angles_shift_bins(:, 2));
shift_bins    = angles_shift_bins(selected_angle_index, 1);
tar_win_start = angles_shift_bins(selected_angle_index, 3);
angle_bin     = angle_bin_range(1) + selected_angle_index - 1;

%% Apply the radar-wearable time shift to all wearable timestamps
shift_seconds = shift_bins / cfg.fs;

time_axis_scg_s  = time_axis_scg  + shift_seconds;
time_axis_ppg_s  = time_axis_ppg  + shift_seconds;
time_axis_bcg_s  = time_axis_bcg  + shift_seconds;
time_axis_neck_s = time_axis_neck + shift_seconds;

peak_time_scg  = peak_time_scg_op  + shift_seconds;
peak_time_ppg  = peak_time_ppg_op  + shift_seconds;
peak_time_bcg  = peak_time_bcg_op  + shift_seconds;
peak_time_neck = peak_time_neck_op + shift_seconds;
