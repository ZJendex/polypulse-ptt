function [bestValue, maxCount] = most_appear_number(sortedArray, threshold)
% most_appear_number  Find the integer with the densest neighbourhood.
%
%   [bestValue, maxCount] = most_appear_number(sortedArray, threshold)
%
%   Slides a counting window of half-width (threshold+1) across all integer
%   values in the range of sortedArray and returns the centre value whose
%   window contains the most elements.
%
%   Inputs:
%     sortedArray - Sorted numeric vector.
%     threshold   - Half-width of the counting window (default: 1).
%
%   Outputs:
%     bestValue - Centre of the densest window.
%     maxCount  - Number of elements in that window.

    if nargin < 2
        threshold = 1;
    end

    maxCount  = 0;
    bestValue = 0;

    for i = sortedArray(1):sortedArray(end)
        count = sum(sortedArray > i - threshold - 1 & sortedArray < i + threshold + 1);
        if count > maxCount
            maxCount  = count;
            bestValue = i;
        end
    end
end
