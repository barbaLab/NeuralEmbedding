function [E,C,Corr] = reduce(D,pars)
        % CCAREDUCE Internal function for CCA

        % Agglomerate all of the conditions, and perform PCA
        E_ = cell(1,pars.nArea);
        C = cell(size(D));
        Corr = cell(size(D));

        if length(D) < 2 && ~iscell(D)
            error('Input must be a cell-array of at least two elements with data from two distinct areas.')
        end
        dims = pars.numPC;
        % endLeg_range = pars.endLeg_range;
        % interest_range = pars.interest_range;


        newD = D;
        if pars.nArea  == 2
            [A,B,Corr,E_{1},E_{2}] = canoncorr(D{:});
            C = {A(:,1:dims),B(:,1:dims)};
        else
            data = cat(1,D{:})';
            d = cellfun(@(x)size(x,1),D);
            [~,Corr,C] = embedding.CCA.mcca(data,d);
        end

        % [U,V] = checkFlip(D{:},C{:},endLeg_range, interest_range);

        % For each condition, store the reduced version of each data vector
        E = cell(pars.nTrial,pars.nArea);
        for ii = 1:pars.nTrial
            index = 0;
            for jj = 1:pars.nArea
                E{ii,jj} = E_{jj}(index + (1:pars.TrialL),1:dims)';
                index = index + pars.TrialL;
            end
        end

        function [U,V] = checkFlip(X,Y,A,B,endLeg_range, interest_range)

            %{
DESCRIPTION: standardized means to determine if weights need to be flipped 
(i.e., if corresponding traces need to be flipped). This will be done by
taking the average firing rate of the resulting top CV (or CCA trace) at 
it ends and seeing if the average firing rate around the region of interest
is lower than it. If it is, then flip the weights.

INPUT: ______________________________________________________
RFA_unitData
S1_unitData
RFA_CCAweights
S1_CCAweights

endleg_range: 1x4 vector detailing 2 pairs of time stamps that define
'baseline' time range

interest_range: 1x4 vector detailing 2 pairs of time stamps that define
time range of interest (e.g., around the time of task onset)

OUTPUT: _____________________________________________________
U & V: corresponding sets of canonical vectors if flip_Status = 0.
            %}

            flipRFA = 0;
            flipS1 = 0;


            % Center the variables
            X = X - mean(X,1);
            Y = Y - mean(Y,1);

            % project onto canonical space
            U = X * A;
            V = Y * B;

%             U = reshape(U, binNum, [], size(U,2));
%             V = reshape(V, binNum, [], size(V,2));

            %{
subplot(1,2,1)
plot(U(:,:,1),'b')
hold on
plot(mean(U(:,:,1),2),'r', LineWidth=5)
xline(50,'r', LineWidth=5)
subplot(1,2,2)
plot(V(:,:,1),'b')
hold on
plot(mean(V(:,:,1),2),'r', LineWidth=5)
xline(50,'r', LineWidth=5)
            %}


            %Flip canonical weights and rerun CCA
            if mean(mean(mean(U([endLeg_range(1):endLeg_range(2),endLeg_range(3):endLeg_range(4)],:,1),2),1)) > mean(mean(mean(U(interest_range(1):interest_range(2),:,1),2),1))
                flipRFA = 1;
                A = -A;
            end
            if mean(mean(mean(V([endLeg_range(1):endLeg_range(2),endLeg_range(3):endLeg_range(4)],:,1),2),1)) > mean(mean(mean(V(interest_range(1):interest_range(2),:,1),2),1))
                flipS1 = 1;
                B = -B;
            end

            if flipRFA || flipS1
                U = X * A;
                V = Y * B;
                
%                 U = reshape(U, binNum, [], size(U,2));
%                 V = reshape(V, binNum, [], size(V,2));
            end
        end

    end