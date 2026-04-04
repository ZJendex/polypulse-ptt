function secondDerivative = computeSecondDerivative(signal, h)
% computeSecondDerivative  Seven-point stencil second derivative.
%
%   secondDerivative = computeSecondDerivative(signal, h)
%
%   Uses a seven-point finite-difference stencil for improved noise
%   robustness. The first 3 and last 3 samples are set to zero.
%
%   Inputs:
%     signal - Input signal vector.
%     h      - Sample spacing (1/fs).
%
%   Output:
%     secondDerivative - Approximated second derivative (same size as signal).

    secondDerivative = zeros(size(signal));
    for i = 4:length(signal)-3
        secondDerivative(i) = (4*signal(i) + (signal(i+1) + signal(i-1)) ...
            - 2*(signal(i+2) + signal(i-2)) - (signal(i+3) + signal(i-3))) / (16*h^2);
    end
end
