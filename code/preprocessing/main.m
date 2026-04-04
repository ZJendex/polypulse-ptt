% main.m  PolyPulse preprocessing pipeline entry point.
%
% Executes the four preprocessing stages in order:
%
%   Stage 1 - Wearable Sensor Synchronization (wearable_sensor_sync.m)
%       Loads raw IMU/PPG data from four wearable sensors, synchronises
%       their clocks via cross-correlation during a calibration phase,
%       detects cardiac peaks (SCG, PPG, BCG, neck pulse) per heartbeat,
%       and computes initial wearable-based PTT estimates.
%
%   Stage 2 - Radar-Wearable Synchronization (radar_sensor_sync.m)
%       Loads radar data, estimates the time offset between the radar and
%       wearable clocks using motion correlation during calibration, and
%       shifts all wearable peak timestamps to the radar time base.
%
%   Stage 3 - Ground-Truth Export (wearable_data_format.m)
%       Crops the synchronised wearable signals to the analysis window and
%       saves ground-truth cardiac peak indices for each modality.
%
%   Stage 4 - Radar Bin Selection (radar_find_bin.m)
%       Identifies the radar range-angle bins that best capture the cardiac
%       signature at each body site (heart, wrist, head, neck) and saves
%       the corresponding radar data snippets.

clear all; close all;
addpath('utils/');

wearable_sensor_sync;
radar_sensor_sync;
wearable_data_format;
radar_find_bin;
