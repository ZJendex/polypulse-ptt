function [candidateHistory, bestBin, fs] = find_bin(imgs, objDistance, ...
        cutoffStart, cutoffEnd, varargin)
% find_bin  Search radar range-angle bins for the strongest cardiac signature.
%
%   [candidateHistory, bestBin, fs] = find_bin(imgs, objDistance, ...
%       cutoffStart, cutoffEnd, Name, Value, ...)
%
%   For each range-angle bin near the expected body-site distance, the
%   function bandpass-filters the phase signal, computes its normalised
%   autocorrelation, and checks whether a peak exists near the expected
%   heart-rate period.  Candidates are collected and ranked by correlation
%   peak height.
%
%   Inputs:
%     imgs         - 3-D radar data cube [frames x range x angle].
%     objDistance   - Expected distance from radar to body site (m).
%     cutoffStart  - First frame index to analyse.
%     cutoffEnd    - Last frame index to analyse.
%
%   Name-Value Options:
%     'thresholdOnBpm'       - BPM tolerance around targetBpm (default: 30).
%     'targetBpm'            - Expected heart rate in BPM (default: 60).
%     'targetAngleDegree'    - Expected azimuth angle in degrees (default: 0).
%     'rangeThresholdIndex'  - Range-bin search half-width (default: 2).
%     'angleThresholdDegree' - Angular search half-width in degrees (default: 180).
%     'secondDerivative'     - Apply 2nd derivative before autocorrelation (default: false).
%     'filter_range'         - [fLow fHigh] bandpass in Hz (default: [0.5 100]).
%     'flipped'              - Negate the signal before processing (default: false).
%
%   Outputs:
%     candidateHistory - [M x 8] sorted candidate matrix.
%         Col 1: autocorrelation peak value
%         Col 2: signal magnitude (energy proxy)
%         Col 3: range bin index
%         Col 4: angle bin index
%         Col 5: best lag (samples) at the cardiac period
%         Col 6: physical angle of the bin (degrees)
%         Col 7-8: reserved (zero)
%     bestBin - [rangeBin, angleBin] of the top-ranked candidate.
%     fs      - Radar frame rate (Hz).

    %% Radar configuration
    cfg = getRadarConfig(imgs);
    fs  = cfg.fs;

    %% Parse optional arguments
    p = inputParser;
    addParameter(p, 'thresholdOnBpm',       30);
    addParameter(p, 'targetBpm',            60);
    addParameter(p, 'targetAngleDegree',    0);
    addParameter(p, 'rangeThresholdIndex',  2);
    addParameter(p, 'angleThresholdDegree', 180);
    addParameter(p, 'secondDerivative',     false);
    addParameter(p, 'filter_range',         [0.5, 100]);
    addParameter(p, 'flipped',              false);
    parse(p, varargin{:});
    opts = p.Results;

    %% Convert BPM tolerance to autocorrelation lag bounds (in samples)
    bpmPeriodUpper = 60 / (opts.targetBpm + opts.thresholdOnBpm);
    bpmPeriodLower = 60 / (opts.targetBpm - opts.thresholdOnBpm);

    %% Crop temporal window
    if ~opts.flipped
        imgsCrop = imgs(cutoffStart:cutoffEnd, :, :);
    else
        imgsCrop = -imgs(cutoffStart:cutoffEnd, :, :);
    end

    %% Determine range-bin search window from target distance
    calibrationOffset = 3;  % empirical bin offset
    targetRangeBin = round(objDistance / cfg.rangeResolution) + calibrationOffset;
    rangeStart = max(2, targetRangeBin - opts.rangeThresholdIndex);
    rangeEnd   = min(cfg.RANGE_FFT, targetRangeBin + opts.rangeThresholdIndex);

    %% Scan all candidate range-angle bins
    candidateHistory = zeros(cfg.numFrames, 8);
    m = 1;

    for rangeBin = rangeStart:rangeEnd
        for angleBin = 1:cfg.angleNFFT
            % Physical azimuth angle of this bin
            binAngle = rad2deg(angle(cfg.xAxis(rangeBin, angleBin) + ...
                                     cfg.yAxis(rangeBin, angleBin) * 1i));

            % Skip bins outside the angular search window
            if binAngle > opts.targetAngleDegree + opts.angleThresholdDegree || ...
               binAngle < opts.targetAngleDegree - opts.angleThresholdDegree
                continue
            end

            binSignal = imgsCrop(:, rangeBin, angleBin);

            % Average magnitude over a 2-second window near the middle of
            % the full recording (energy proxy for candidate ranking)
            midStart  = round(size(imgs, 1) / 2);
            magnitude = sum(abs(imgs(midStart:midStart + 2*round(fs), rangeBin, angleBin))) / (2 * fs);

            % Bandpass filter and phase extraction
            processedSignal = signalProcessingBasic(binSignal, fs, ...
                'filter_range', opts.filter_range, 'flipped', false);
            if opts.secondDerivative
                processedSignal = computeSecondDerivative(processedSignal, 1/fs);
            end

            %% Autocorrelation-based cardiac periodicity check
            [acf, lags] = xcorr(processedSignal, processedSignal, 'normalized');
            lagStartIdx = ceil(length(acf)/2) + round(bpmPeriodUpper * fs);
            lagEndIdx   = ceil(length(acf)/2) + round(bpmPeriodLower * fs);
            targetAcf   = acf(lagStartIdx:lagEndIdx);

            [acfPeaks, peakLagIdx] = findpeaks(targetAcf, 'MinPeakWidth', 5);
            if isempty(acfPeaks)
                continue
            end

            [maxAcf, maxIdx] = max(acfPeaks);
            peakLags = lags(lagStartIdx:lagEndIdx);
            peakLags = peakLags(peakLagIdx);
            bestLag  = peakLags(maxIdx);

            if maxAcf > -10
                candidateHistory(m, 1) = maxAcf;
                candidateHistory(m, 2) = magnitude;
                candidateHistory(m, 3) = rangeBin;
                candidateHistory(m, 4) = angleBin;
                candidateHistory(m, 5) = bestLag;
                candidateHistory(m, 6) = binAngle;
                m = m + 1;
            end
        end
    end

    %% Sort candidates by autocorrelation peak (descending)
    candidateHistory = sortrows(candidateHistory, -1);
    bestBin = [candidateHistory(1, 3), candidateHistory(1, 4)];
end
