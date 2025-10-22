function [C,Corr] = reduce(D_all,pars)
        % CCAREDUCE Internal function for CCA
        nSets = length(D_all);

        % Agglomerate all the sets, and perform mCCA

        % 
        % if length(D) < 2 && ~iscell(D)
        %     error('Input must be a cell-array of at least two elements with data from two distinct areas.')
        % end
        % endLeg_range = pars.endLeg_range;
        % interest_range = pars.interest_range;
        
        data = cellfun(@(eall)cat(2,eall{:}),D_all,'UniformOutput',false);
        d = cellfun(@(x)size(x,1),data);
        [~,Corr,C] = embedding.CCA.mcca(cat(1,data{:})',d);

    end