function [dist, t] = plot_waveformOnBin(binIdx, imgs, cutoffStart, cutoffEnd, varargin)
% plot_waveformOnBin  Extract, process, and optionally plot a radar bin waveform.
%
%   [dist, t] = plot_waveformOnBin(binIdx, imgs, cutoffStart, cutoffEnd, ...)
%
%   Extracts the complex signal from a single range-angle bin, applies
%   phase extraction and bandpass filtering, converts to displacement
%   in millimetres, and optionally generates a plot.
%
%   Inputs:
%     binIdx      - [rangeBin, angleBin] indices into the data cube.
%     imgs        - 3-D radar data cube [frames x range x angle].
%     cutoffStart - First frame index (0 = use all frames).
%     cutoffEnd   - Last frame index.
%
%   Name-Value Options:
%     'rawX'         - Use sample index for x-axis (default: false).
%     'filter'       - Apply bandpass filter (default: true).
%     'flipped'      - Negate signal (default: false).
%     'phase'        - Phase extraction; false = log-magnitude (default: true).
%     'filter_range' - [fLow fHigh] Hz (default: [0.5 100]).
%     'emd_i'        - EMD IMF range (default: [-1 -1] = off).
%     'plot'         - Generate plot (default: true).
%
%   Outputs:
%     dist - Displacement waveform (mm).
%     t    - Time vector (s).

    cfg = getRadarConfig(imgs);

    p = inputParser;
    addParameter(p, 'rawX',         false);
    addParameter(p, 'filter',       true);
    addParameter(p, 'flipped',      false);
    addParameter(p, 'phase',        true);
    addParameter(p, 'filter_range', [0.5, 100]);
    addParameter(p, 'emd_i',        [-1, -1]);
    addParameter(p, 'plot',         true);
    parse(p, varargin{:});
    opts = p.Results;

    %% Extract bin data
    if nargin > 3 && cutoffStart > 0
        binData = imgs(cutoffStart:cutoffEnd, binIdx(1), binIdx(2));
    else
        binData = imgs(:, binIdx(1), binIdx(2));
    end

    %% Signal processing (phase extraction + filtering)
    processed = signalProcessingBasic(binData, cfg.fs, ...
        'filter', opts.filter, 'phase', opts.phase, ...
        'filter_range', opts.filter_range, 'flipped', opts.flipped, ...
        'emd_i', opts.emd_i);

    %% Convert phase to displacement (mm)
    dist = processed * cfg.c / (4 * pi * cfg.Fc) * 1000;
    t    = (1:length(processed)) / cfg.fs;

    %% Optional plot
    if opts.plot
        if opts.rawX
            plot(dist);
            xlabel('Samples');
        else
            plot(t, dist);
            xlabel('Time (s)');
        end
        ylabel('Displacement (mm)');

        binAngle   = rad2deg(angle(cfg.xAxis(binIdx(1), binIdx(2)) + ...
                                   cfg.yAxis(binIdx(1), binIdx(2)) * 1i));
        rangeMeter = binIdx(1) * cfg.rangeResolution;
        if opts.phase
            modeStr = 'Phase';
        else
            modeStr = 'Magnitude';
        end
        title(sprintf('%s | Range bin %d, Angle bin %d | fs %d Hz | Dist %.2f m, Angle %.1f deg', ...
            modeStr, binIdx(1), binIdx(2), cfg.fs, rangeMeter, binAngle));
    end
end
