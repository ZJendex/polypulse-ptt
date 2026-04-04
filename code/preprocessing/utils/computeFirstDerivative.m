function firstDerivative = computeFirstDerivative(signal, h)
% computeFirstDerivative  Central finite-difference first derivative.
%
%   firstDerivative = computeFirstDerivative(signal, h)
%
%   Uses a central difference scheme. The first 3 and last 3 samples are
%   set to zero (boundary padding).
%
%   Inputs:
%     signal - Input signal vector.
%     h      - Sample spacing (1/fs).
%
%   Output:
%     firstDerivative - Approximated first derivative (same size as signal).

    firstDerivative = zeros(size(signal));
    for i = 4:length(signal)-3
        firstDerivative(i) = (signal(i+1) - signal(i-1)) / (2*h);
    end
end
