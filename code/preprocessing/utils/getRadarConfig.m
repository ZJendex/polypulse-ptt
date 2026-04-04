function cfg = getRadarConfig(imgs)
% getRadarConfig  Compute radar system parameters from TI mmWave Studio config.
%
%   cfg = getRadarConfig(imgs)
%
%   Input:
%     imgs - 3-D radar data cube [frames x range_bins x angle_bins].
%            Only used to determine the number of frames.
%
%   Output:
%     cfg  - struct with fields: c, Fc, fs, rangeResolution, RANGE_FFT,
%            angleNFFT, numFrames, xAxis, yAxis.

    %% TI mmWave Studio hardware parameters
    numADCSample      = 128;
    adcSampleRate     = 2.28e6;       % Hz
    startFreqConst    = 7.70e10;      % Hz
    chirpSlope        = 6.3343e13;    % Hz/s
    chirpIdleTime     = 9.70e-5;      % s
    adcStartTimeConst = 6.00e-6;      % s
    chirpRampEndTime  = 6.314e-5;     % s
    framePeriodicity  = 0.1;          % s  (frame repetition interval)
    nchirpLoops       = 50;

    %% Physical constants
    cfg.c  = 3e8;      % speed of light (m/s)
    cfg.Fc = 77e9;     % carrier frequency (Hz)

    %% Derived RF parameters
    chirpRampTime   = numADCSample / adcSampleRate;
    chirpBandwidth  = chirpSlope * chirpRampTime;
    cfg.rangeResolution = cfg.c / (2 * chirpBandwidth);

    %% FFT and axis sizes
    cfg.RANGE_FFT = numADCSample;   % 128 range bins
    AZIM_FFT      = 128;            % angle FFT bins
    cfg.angleNFFT = AZIM_FFT;
    MAX_RANGE     = cfg.RANGE_FFT * cfg.rangeResolution;

    %% Effective frame rate
    cfg.fs = (1 / framePeriodicity) * nchirpLoops;  % Hz

    %% Number of frames from data
    cfg.numFrames = size(imgs, 1);

    %% Cartesian coordinate grids for the range-angle map
    sineTheta = -2 * linspace(-0.5, 0.5, AZIM_FFT);
    cosTheta  = sqrt(1 - sineTheta.^2);
    [rangeGrid, sineMat] = ndgrid(linspace(0, MAX_RANGE, cfg.RANGE_FFT), sineTheta);
    [~,         cosMat]  = ndgrid(linspace(0, MAX_RANGE, cfg.RANGE_FFT), cosTheta);
    cfg.xAxis = rangeGrid .* cosMat;
    cfg.yAxis = rangeGrid .* sineMat;
end
