function [t, signal, outlierIdx] = outlierRemovalForDataPairs(t, signal)
% outlierRemovalForDataPairs  Remove spike outliers from paired time-signal data.
%
%   [t, signal, outlierIdx] = outlierRemovalForDataPairs(t, signal)
%
%   Detects points whose forward or backward difference exceeds
%   NUM_STD_THRESH * std(diff) in either the time or signal channel,
%   then removes those points from both arrays.
%
%   Inputs:
%     t      - Timestamp vector [1 x N].
%     signal - Signal vector [1 x N].
%
%   Outputs:
%     t          - Cleaned timestamps.
%     signal     - Cleaned signal.
%     outlierIdx - Indices of removed points in the original arrays.

    NUM_STD_THRESH = 100;

    diffBefore_t   = [0, abs(diff(t))];
    diffAfter_t    = [abs(diff(t)), 0];
    diffBefore_sig = [0, abs(diff(signal))];
    diffAfter_sig  = [abs(diff(signal)), 0];

    thresh_t   = NUM_STD_THRESH * std(abs(diff(t)));
    thresh_sig = NUM_STD_THRESH * std(abs(diff(signal)));

    idx_t   = find((diffBefore_t > thresh_t)   | (diffAfter_t > thresh_t));
    idx_sig = find((diffBefore_sig > thresh_sig) | (diffAfter_sig > thresh_sig));
    outlierIdx = unique([idx_t, idx_sig]);

    t(outlierIdx)      = [];
    signal(outlierIdx) = [];
end
