
function E = project(D, C)

E = cell(size(D));
alldata = [D{:}];

sc = alldata'*C;

% For each condition, store the reduced version of each data vector
index = 0;
for ii = 1:length(D)
    E{:} = sc(index + (1:size(D{ii},2)),:)';
    index = index + size(D{ii},2);
end %ii
end
