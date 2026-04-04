function [interpolatedData, uniformTime] = interpolationWithTimestamp(originalData, originalTime, fsTarget)
% interpolationWithTimestamp  Resample irregularly-sampled data to a uniform grid.
%
%   [interpolatedData, uniformTime] = interpolationWithTimestamp(originalData, originalTime, fsTarget)
%
%   Uses cubic spline interpolation to resample data from irregular
%   timestamps onto a uniform time grid at the specified sampling rate.
%
%   Inputs:
%     originalData - Signal values at irregular sample times.
%     originalTime - Corresponding timestamps (s).
%     fsTarget     - Desired output sampling rate (Hz).
%
%   Outputs:
%     interpolatedData - Resampled signal on uniform grid.
%     uniformTime      - Uniform time vector at fsTarget.

    startTime = originalTime(1);
    endTime   = originalTime(end);
    uniformTime      = (startTime:1/fsTarget:endTime)';
    interpolatedData = spline(originalTime, originalData, uniformTime);
end
