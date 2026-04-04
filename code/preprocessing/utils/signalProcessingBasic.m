function outputSignal = signalProcessingBasic(data, fs, varargin)
% signalProcessingBasic  Phase-extract, filter, and detrend a radar bin signal.
%
%   outputSignal = signalProcessingBasic(data, fs, Name, Value, ...)
%
%   Inputs:
%     data - Complex radar samples from a single range-angle bin [N x 1].
%     fs   - Sampling rate (Hz).
%
%   Name-Value Options:
%     'filter'       - Apply Butterworth bandpass (default: true).
%     'phase'        - Use phase extraction; false uses log-magnitude (default: true).
%     'filter_range' - [fLow fHigh] passband in Hz (default: [0.5 60]).
%     'flipped'      - Negate signal before processing (default: false).
%     'emd_i'        - [start end] IMF range for EMD decomposition (default: [-1 -1] = off).
%
%   Output:
%     outputSignal - Processed real-valued signal [N x 1].

    p = inputParser;
    addParameter(p, 'filter',       true);
    addParameter(p, 'phase',        true);
    addParameter(p, 'filter_range', [0.5, 60]);
    addParameter(p, 'flipped',      false);
    addParameter(p, 'emd_i',        [-1, -1]);
    parse(p, varargin{:});
    opts = p.Results;

    %% Extract phase or log-magnitude
    if opts.phase
        sig = unwrap(angle(squeeze(data)));
    else
        sig = log10(abs(data));
    end

    %% Optional EMD decomposition
    if opts.emd_i(1) ~= -1
        [imf, ~] = emd(sig);
        sig = sum(imf(:, opts.emd_i(1):opts.emd_i(2)), 2);
    end

    %% Optional signal inversion
    if opts.flipped
        sig = -sig;
    end

    %% Butterworth bandpass filter (applied as cascaded high+low)
    if opts.filter
        [b, a] = butter(2, opts.filter_range(1)/(fs/2), 'high');
        sig = filtfilt(b, a, sig);
        [b, a] = butter(2, opts.filter_range(2)/(fs/2), 'low');
        sig = filtfilt(b, a, sig);
    end

    %% Remove polynomial trend (order 4)
    outputSignal = detrend(sig, 4);
end
