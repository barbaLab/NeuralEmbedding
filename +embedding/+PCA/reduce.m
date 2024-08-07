function [E ,C ,VarExplained] = reduce(D,dims)

        % PCAREDUCE Internal function for PCA
        %   PCAREDUCE(D,DIMS) returns a structure of the same form as D, except
        %   the data has been reduced with PCA. All conditions and trials are
        %   considered together to get the best joint reduction.

        E = cell(size(D));
        C = cell(1);
        VarExplained = cell(1);

        % Agglomerate all of the conditions, and perform PCA
        alldata = [D.data];
        [u,sc,lat] = pca(alldata');

        % For each condition, store the reduced version of each data vector
        index = 0;
        for i=1:length(D)
            D(i).data = sc(index + (1:size(D(i).data,2)),1:dims)';
            index = index + size(D(i).data,2);
        end
        [E{:}] = deal(D.data);
        C{1} = u(:,1:dims);
        VarExplained{1} = cumsum(lat) ./ sum(lat);  % eigenvalues

    end