% wearable_data_format.m  Stage 3 -- Crop synchronised signals and export ground-truth peaks.
%
% Crops all synchronised wearable signals and peak timestamps to the
% analysis window [cali_time, end_cutting_time] and saves ground-truth
% cardiac peak indices for each modality (SCG, PPG, BCG, neck).
%
% Requires workspace variables from Stages 1 and 2:
%   cali_time, end_cutting_time, fs
%   time_axis_scg_s, time_axis_ppg_s, time_axis_bcg_s, time_axis_neck_s
%   scg_data, ppg_data, bcg_data, neck_data
%   peak_time_scg, peak_time_ppg, peak_time_bcg, peak_time_neck
%
% Produces:
%   cut_index  -- [start_sample, end_sample] for the analysis window
%   Saves GT_example.mat with peak sample indices for each modality.

%% Define analysis window in sample indices
cut_index(1, 1) = cali_time * fs;
cut_index(1, 2) = end_cutting_time * fs;

t_start = cut_index(1, 1) / fs;
t_end   = cut_index(1, 2) / fs;

%% Crop each modality to the analysis window

% SCG
cut_time_axis_scg = time_axis_scg_s(time_axis_scg_s > t_start);
cut_time_axis_scg = cut_time_axis_scg(cut_time_axis_scg < t_end);
cut_time_axis_scg = cut_time_axis_scg - cut_time_axis_scg(1);
cut_scg_data = scg_data(time_axis_scg_s > t_start);
cut_scg_data = cut_scg_data(time_axis_scg_s(time_axis_scg_s > t_start) < t_end);
cut_peak_time_scg = peak_time_scg(peak_time_scg > t_start);
cut_peak_time_scg = cut_peak_time_scg(cut_peak_time_scg < t_end);

% PPG
cut_time_axis_ppg = time_axis_ppg_s(time_axis_ppg_s > t_start);
cut_time_axis_ppg = cut_time_axis_ppg(cut_time_axis_ppg < t_end);
cut_ppg_data = ppg_data(time_axis_ppg_s > t_start);
cut_ppg_data = cut_ppg_data(time_axis_ppg_s(time_axis_ppg_s > t_start) < t_end);
cut_time_axis_ppg = cut_time_axis_ppg - cut_time_axis_ppg(1);
cut_peak_time_ppg = peak_time_ppg(peak_time_ppg > t_start);
cut_peak_time_ppg = cut_peak_time_ppg(cut_peak_time_ppg < t_end);

% BCG
cut_time_axis_bcg = time_axis_bcg_s(time_axis_bcg_s > t_start);
cut_time_axis_bcg = cut_time_axis_bcg(cut_time_axis_bcg < t_end);
cut_bcg_data = bcg_data(time_axis_bcg_s > t_start);
cut_bcg_data = cut_bcg_data(time_axis_bcg_s(time_axis_bcg_s > t_start) < t_end);
cut_time_axis_bcg = cut_time_axis_bcg - cut_time_axis_bcg(1);
cut_peak_time_bcg = peak_time_bcg(peak_time_bcg > t_start);
cut_peak_time_bcg = cut_peak_time_bcg(cut_peak_time_bcg < t_end);

% Neck
cut_time_axis_neck = time_axis_neck_s(time_axis_neck_s > t_start);
cut_time_axis_neck = cut_time_axis_neck(cut_time_axis_neck < t_end);
cut_time_axis_neck = cut_time_axis_neck - cut_time_axis_neck(1);
cut_neck_data = neck_data(time_axis_neck_s > t_start);
cut_neck_data = cut_neck_data(time_axis_neck_s(time_axis_neck_s > t_start) < t_end);
cut_peak_time_neck = peak_time_neck(peak_time_neck > t_start);
cut_peak_time_neck = cut_peak_time_neck(cut_peak_time_neck < t_end);

%% Save ground-truth peak indices
% Convert peak timestamps to sample indices relative to the analysis window start
data_name   = sprintf('GT_%s', 'example');
base_folder = fullfile('..', '..', 'data', 'example', 'raw', 'example_user', 'example_recording');
if ~isfolder(base_folder)
    mkdir(base_folder)
end
full_file_path = fullfile(base_folder, data_name);

scg_peaks_gt  = round((cut_peak_time_scg  - cali_time) * fs);
ppg_peaks_gt  = round((cut_peak_time_ppg  - cali_time) * fs);
bcg_peaks_gt  = round((cut_peak_time_bcg  - cali_time) * fs);
neck_peaks_gt = round((cut_peak_time_neck - cali_time) * fs);

save(full_file_path, 'scg_peaks_gt', 'ppg_peaks_gt', 'bcg_peaks_gt', 'neck_peaks_gt');
