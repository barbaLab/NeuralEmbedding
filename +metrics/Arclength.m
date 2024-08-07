function M = Arclength(E,pars)
%% Compute trajectory arclentgh. 
% input
%                       projectedData 1xnTrial cell array, each w nUnitsxT
%                       formatted sparse matrices
% output
%                       M 1x1 median value across trials

% Format data conviently for arclentgh processing
FormttedData = cellfun(@(x)num2cell(x,2),E,'UniformOutput',false);
M = cellfun(@(x) metrics.compute.arclength(x{:}, pars.method),...
            FormttedData);
M = median(M);
