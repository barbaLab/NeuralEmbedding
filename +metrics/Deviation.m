%%
method = 'gpfa';
pars.t = 2e3+(-1000:1000);
pars.Reproject = true;
splitUnits = true;
pars.numPC = 30;
pars.prekern = 0;
pars.postkern = 200;
pars.useGpu = true;
pars.zscore = true;
pars.binWidth = 10;

projectedData = cell(12,5);
C = projectedData;
Latent = C;
for a = 1:12
    D = Dfull.D(a,:);
    for ii=[1:5]
        if isempty(D{ii})
            continue
        end
        pars.splitUnits = [0 find(D{ii}(1).area == 'RFA',1,'last') size(D{ii}(1).data,1) 0 size(D{ii}(1).data,1)];
        if ~splitUnits 
            pars.splitUnits = pars.splitUnits([1 end]);
        end
        [C{a,ii},projectedData{a,ii},Latent{a,ii}] = NeuralEmbedding([D{ii}],method,pars);
        
%         c=c+1;
    end
end


fs = 1000;
distM_ = cell(12,5,numel( pars.splitUnits)-1);
for a = 1:12
    for block = 1:5
        if isempty(projectedData{a,block})
            continue
        end
        for area = 1:numel(pars.splitUnits)-1
            data = cat(3,projectedData{a,block}{area}.data);
            dataM_ = mean(data,3);
            dist = squeeze(sqrt(sum((data - dataM_).^2,1)));
            distAll = sum(dist,1);
            distM_{a,block,area} = (distAll);

        end
    end
end
nAreas = size(distM_,3);


fs = 1000;
for a = 1:12
    for block = 1:5
        if isempty(projectedDataAll{a,block})
            continue
        end
        for area = 1
            data = cat(3,projectedDataAll{a,block}{area}.data);
            dataM_ = mean(data,3);
            dist = squeeze(sqrt(sum((data - dataM_).^2,1)));
            distAll = sum(dist,1);
            distM_{a,block,3} = (distAll);

        end
    end
end

%%
[success,fail,noAtt,nAn,nBlck] = getSuccessRate(tankObj);
succ = success./(success+fail);
succ = repmat(succ,1,1,nAreas);

Deficit = false(12,5,nAreas);
for a = 1:12
    for block = 1:5
        deltaScore = max(succ(a,1:3)) - min(succ(a,4:5));
        if deltaScore > .3
            Deficit(a,4:5,:) = true;
        end
    end
end

%%
clf;
bl = repmat(ones(12,1)*[1:5],1,1,nAreas);
an = (1:12)'*ones(1,5);
an = repmat(an,1,1,nAreas);
% area = cat(3,repmat({'RFA'},12,5),...
%     repmat({'S1'},12,5),...
%     repmat({'All'},12,5));
area = cat(3,repmat({'RFA'},12,5),...
    repmat({'S1'},12,5));

g = gramm('x',bl(:),'y',distM_(:),'color',an(:),'column',area(:));
g.set_names('y','Trajectory deviation from the mean','x','Day','color','Animal');
g.set_point_options('markers',{'o','^'},'base_size',7);
% g.geom_abline('intercept',1);
g.stat_glm('geom','line','disp_fit',false,'fullrange',true);
% g.set_line_options('styles',{'--'},'base_size',.8);
% g.update('marker',Deficit(:));
g.geom_point('alpha',.8);
g.draw;


%%
data = [];
an = [];
bl = [];
for aa = 1:12
    for bb= 1:5
    data = [data distM_{aa,bb,3}];
    an = [an aa*ones(size(distM_{aa,bb,3}))];
    bl = [bl bb*ones(size(distM_{aa,bb,3}))];
    end
end
    for aa=1:12
        figure
    g = gramm('x',data,'color',bl,'subset',an==aa,'column',bl);
    g.stat_bin('nbins',50 ,'geom','overlaid_bar','normalization','probability');
    g.draw()
    end

%% Divide blocks based on condition type

methods = {'PCA'};
scale = [1 10];
for m = 1:numel(methods)
    method = methods{m};

%   load([method Areas_Personalized.mat]);
    load([method '_All.mat']);
    projectedDataAll = AllProjected;
    path = '\\wsl.localhost\Ubuntu-20.04\home\fede\DataAnalysis\R03';
    areas = {'RFA','S1 ','.  '};
    for aa = 1:numel(projectedDataAll)
        if isempty(projectedDataAll{aa}),continue;end

        conditions = {projectedDataAll{aa}{1}.condition};

        Blocks = unique(...
            cellfun(@(x)x(end-7:end),...
            unique(conditions),'UniformOutput',false));
        name = sprintf('%s%s.xlsx',method,tankObj.Children(aa).Name);
        fprintf('\nExporting %s.%s.%s',tankObj.Children(aa).Name,Blocks{1},areas{1});

        for bb = 1:numel(Blocks)
            for ar = 1:numel(areas)
                eraser = repmat('\b',1,16);
                fprintf([eraser '%s.%s.%s'],tankObj.Children(aa).Name,Blocks{bb},areas{ar});
                idx = find(contains(conditions,Blocks{bb}));
                data = cat(1,projectedDataAll{aa}{ar}(idx).data);
                writematrix(data,fullfile(path,deblank(areas{ar}),name),...
                    'Sheet',Blocks{bb},'WriteMode','overwritesheet');
            end % ar
        end % ii
    end % aa
end
%%
projectedDataAll = cell(12,1);
for aa = 1:12
    tmp = cat(1,projectedData{aa,:});
    for ii = 1:size(tmp,2)
        projectedDataAll{aa}{ii} = cat(2,tmp{:,ii});
    end
end
projectedDataAll{cellfun(@isempty,projectedDataAll)} = cell(size(projectedDataAll{1}));
%%
methods = {'GPFA'};
scale = [1 10];
for m = 1:numel(methods)
    method = methods{m};

    % 
    % load([method 'All.mat']);
    % projectedDataAll = projectedData;
    path = '\\wsl.localhost\Ubuntu-20.04\home\fede\DataAnalysis\R03';
    areas = {'RFA','S1',''};
    for aa = 1:numel(projectedDataAll)
        if isempty(projectedDataAll{aa}),continue;end

        conditions = {projectedDataAll{aa}{1}.condition};

        Blocks = unique(...
            cellfun(@(x)x(end-7:end),...
            unique(conditions),'UniformOutput',false));
        name = sprintf('%s%s.xlsx',method,tankObj.Children(aa).Name);
        fprintf('\nExporting %s.%s.%s',tankObj.Children(aa).Name,Blocks{1},areas{1});

        for bb = 1:numel(Blocks)
            for ar = 1:numel(areas)
                eraser = repmat('\b',1,16);
                fprintf([eraser '%s.%s.%s'],tankObj.Children(aa).Name,Blocks{bb},areas{ar});
                idx = contains(conditions,Blocks{bb});
                data = cat(1,projectedDataAll{aa}{ar}(idx).data);
                writematrix(binData(data,1)*scale(m),fullfile(path,deblank(areas{ar}),name),...
                    'Sheet',Blocks{bb},'WriteMode','overwritesheet');
            end % ar
        end % ii
    end % aa
end

%%
method = 'GPFA';
bNameList = {'preLesion1','preLesion2','preLesion3','postLesion1','postLesion2'};
path = '\\wsl.localhost\Ubuntu-20.04\home\fede\DataAnalysis\R03';
for aa = 1:12
    name = sprintf('%s%s.xlsx',method,tankObj.Children(aa).Name);
    for ii = 1:5
        if isempty(projectedDataAll{aa,ii})
            continue;
        end
        writematrix(cat(1,projectedDataAll{aa,ii}{1}.data),fullfile(path,name),...
            'Sheet',bNameList{ii})
    end
end 

%%
[success,fail,noAtt,nAn,nBlck] = getSuccessRate(tankObj);
succ = success./(success+fail);

deficit = (max(succ(:,1:3),[],2)- min(succ(:,4:5),[],2))>.3;
    bNameList = {'preLesion1','preLesion2','preLesion3','postLesion1','postLesion2'};
    Res = cell(1,12);
    method = 'GPFA';
    ar=0;
    for area = {'RFA','S1','.'}
        ar = ar+1;
        tabname = sprintf('Results%s%s.xlsx',area{:},method);
        tabname = fullfile('\\wsl.localhost\Ubuntu-20.04\home\fede\DataAnalysis','R03','Res',tabname);
        for aa=1:numel(tankObj.Children)
            if ~ismember(tankObj.Children(aa).Name,sheetnames(tabname))
                disp('Skipped')
                continue;
            end
            Res{aa} = readtable(tabname,'Sheet',tankObj.Children(aa).Name,'ReadVariableNames',true,...
                'ReadRowNames',true);
        end
        anNames = {tankObj.Children.Name};
        blNames = {'preLesion1','preLesion2','preLesion3','postLesion1','postLesion2'};
    
        data = [];
        an = [];
        bl = [];
        sr = [];
        def = [];
        T = [];
        blLabel = {[1 2 4 5],[1 2 4 5],1:5,1:5,1:5,1:5,1:3,1:5,[1 4],[1 4],[1 4 5],[1 4 5]};
        for aa = 1:size(Res,2)
            for bb= 1:size(Res{aa},1)
                data = [data Res{aa}{bb,:}];
                an = [an repmat(anNames(aa),size(Res{aa}{bb,:}))];
                bl = [bl repmat(blNames(blLabel{aa}(bb)),size(Res{aa}{bb,:}))];
                sr = [sr repmat(succ(aa,blLabel{aa}(bb)),size(Res{aa}{bb,:}))];

                if any(blLabel{aa}(bb) == [4:5]) && deficit(aa)
                    def = [def true(size(Res{aa}{bb,:}))];
                else
                    def = [def false(size(Res{aa}{bb,:}))];
                end
            end
        end
        AreaRes(ar).def = def;
        AreaRes(ar).data = data;
        AreaRes(ar).an = an;
        AreaRes(ar).bl = bl;
        AreaRes(ar).area = repmat(area,size(bl));
        AreaRes(ar).sr = sr;
    end

% for aa=anNames
%     figure
%     g = gramm('x',data,'color',bl,'subset',strcmp(an,aa));
%     g.facet_wrap(an);
%     g.stat_bin('nbins',50 ,'geom','bar','normalization','probability');
%     g.set_names("color",'')
%     g.draw()
% end

figure
g = gramm('x',[AreaRes.data],'color',[AreaRes.bl],'fig',[AreaRes.area]);
g.stat_bin('nbins',50 ,'geom','line','normalization','probability');
g.facet_wrap([AreaRes.an]);
% g.set_names('column','Animal');
g.draw()

Ar = [AreaRes.area];
Ar(strcmp(Ar,'.'))={'All Units'};
An = cellfun(@(a)find(strcmp(a,anNames)),[AreaRes.an]);

figure("Position",[102.6000 339.4000 668 328.8000]);
g2 = gramm('y',[AreaRes.data],'color',An,'x',[AreaRes.sr],'column',Ar);
g2.geom_polygon('x',{[-1 .4 .4 -1]},'y',{[0 0 3 3]});
g2.stat_glm('geom','line','fullrange',false)
g2.stat_summary('geom','point')
g2.set_point_options('base_size',7);
g2.set_names('x','Success rate','y','Trajectory dispersion','Color','Rats','column','');
g2.set_color_options('map','brewer3');
g2.set_line_options("base_size",2.5);
% g2.axe_property('ylim',[1.5 2.4],'xlim',[-.05 1.05]);
g2.draw

bl = cellfun(@(l)strrep(l,'Lesion',''),[AreaRes.bl] ,'UniformOutput',false);
T=cellfun(@(i)contains([AreaRes.bl],i),...
    {'pre','post'},'UniformOutput',false);
T = [2 4.5]*cat(1,T{:});


figure("Position",[102.6000 339.4000 668 328.8000]);
g3 = gramm('y',[AreaRes.data],'color',An,'column',Ar,...
    'x',bl,'marker',double([AreaRes.def]));
g3.geom_polygon('x',{[3.5 6 6 3.5]},'y',{[0 0 3 3]});
g3.stat_summary('geom','point');
g3.set_order_options('x',{'pre1','pre2','pre3','post1','post2'});
g3.set_point_options('base_size',7);
g3.set_names('x','Day','y','Trajectory dispersion','Color','Rats','column','');
g3.set_color_options('map','brewer3');
g3.draw;
g3.update('x',T,'marker',[]);
g3.stat_summary('geom','line');
g3.set_line_options("base_size",2.5);
g3.axe_property('ylim',[1.5 2.4]);
g3.draw();

%% tubeplot
pars.binWidth = 1;
% (integer, seconds) --- bins the data (assuming 1ms time steps)
pars.kern = 50; %ms
pars.t = (-100:100)+2e2;
uAreas = {'all_units'};
method = 'umap';
for aa = 1:numel(tankObj.Children)
    %%
    an = tankObj{aa};
    allNewD = projectedData(aa,:);
    %%
    data = cell(size(allNewD{1}));
    cidx = data;
    
    for uu = 1:size(allNewD,2)
        for ar = 1:numel(allNewD{uu})
            data_ = zeros([size(allNewD{uu}{ar}(1).data(:,pars.t)) size(allNewD{uu}{ar},2)]);
            cidx_ = zeros(size(allNewD{uu}{ar},2),1);
    
            for ii=1:size(allNewD{uu}{ar},2)
                data_(:,:,ii) = smoother(allNewD{uu}{ar}(ii).data(:,pars.t),pars.kern,pars.binWidth);
                cidx_(ii) = find(strcmp(allNewD{uu}{ar}(ii).condition,conditions));
            end
            data{ar} = cat(3,data{ar},data_);
            cidx{ar} = cat(1,cidx{ar},cidx_);
        end
    end
    
    
    IEvI = repmat({cat(1,IEvI0{1:end})},1,numel(allNewD{uu}));
    
    clear dataRFA dataS1 D_ cidxRFA cidxS1 data_ cidx_ IEvI0;
    
    
    bl = an{1};
    
    
    conditions = [{'FailIntact_1'},    {'FailIntact_2'},    {'FailIntact_3'},...        1                   2                   3
                 {'FailLesion_1'},    {'FailLesion_2'},...                              4                   5            
                 {'SuccessIntact_1'},    {'SuccessIntact_2'},   {'SuccessIntact_3'},... 6                   7                   8
                 {'SuccessLesion_1'},    {'SuccessLesion_2'}];                      %   9                   10
    
    % retrieve events
    TargetEvts = {'ReachStarted','Contact','Retract'};
    Alignment = 'GraspStarted';
    IEvI0 = cell(size(an.Children));    
    for b = 1:numel(an.Children)
        bb = an{b};
        query = {{'Name','Stereotyped','Data',1},...
            {'Name',Alignment}};
        reEvt = bb.filterEvt(query{:});
        IEvI0{b} = zeros(numel(reEvt),numel(TargetEvts));
        for ee = 1:numel(TargetEvts)
            query = {{'Name','Stereotyped','Data',1},...
                {'Name',TargetEvts{ee}}};
            thisEvt = bb.filterEvt(query{:});
    
            idx1 = ismember([thisEvt.Trial],[reEvt.Trial]);
            idx2 = ismember([reEvt.Trial],[thisEvt.Trial]);
            IEvI0{b}(idx2,ee) = round(([thisEvt(idx1).Ts]-[reEvt(idx2).Ts])*1e3);
        end
        IEvI0{b}(:,end+1) = round(...
            (bb.Trial([reEvt.Trial],2) - [reEvt.Ts]') * 1e3 );
    end
    TargetEvts{end+1} = 'Door closed';
    
    ncurves = 30;
    conds = {[1 6] [2 7] [3 8] [4 9] [5 10]};
    map = {'blues','BuGn','greens','reds','purples'};
    meanW = 3;
    meanC = [0 0 0];
    meanMS = 150;
    
    trW = 1;
    trAlpha = .95;
    trajColor = [0.4784 0.5333 0.5569]+.3;
    
    MeEdAlpha = .35;
    MeFaAlpha = 0;
    
    MS =25;
    
    nvertex = 10;
    thMergePOints = 1e-6;
    pc = 1:3;
    
    for ar = 1:numel(data)
    
        for ii = 1:length(conds)
    
            idx = find(ismember(cidx{ar},conds{ii})); % select trials in the right condition
            if isempty(idx)
                continue;
            end
    
            cm = getR03cmap(conditions{conds{ii}(1)},256);
            f(ii,ar) = figure('Position', [2,47,1087,946]);
            ax(ii) = axes(f(ii,ar));%nexttile;
    
            tit = sprintf('%s %s %s',tankObj.Children(aa).Name,uAreas{ar},...
                strjoin(conditions(conds{ii}),'&'));
            [M{ii,ar},mLine{ii,ar},traj{ii,ar},T{ii,ar},C{ii,ar}] = ...
                plotTube(data{ar}(pc,:,idx),nvertex,thMergePOints,cm,ncurves,...
                MeEdAlpha,MeFaAlpha,...
                meanC,meanMS,meanW,...
                trAlpha,trW,MS,trajColor,...
                tit,...
                IEvI{ar}(idx,:),{'square','diamond','hexagram','Pentagram'},...
                strrep([TargetEvts {Alignment}],'Started',''));
            xlabel(ax(ii),sprintf('%s %d',upper(method),pc(1)));
            ylabel(ax(ii),sprintf('%s %d',upper(method),pc(2)));
            zlabel(ax(ii),sprintf('%s %d',upper(method),pc(3)));
            %         view(ax(ii),110,20);
        end
        %     lp(ar) = linkprop(ax,{'View'});
        %     legend(cellfun(@(x) x(1),me),conditions(conds))
    end
end % aa

function [y_bin,t] = binData(X,bin_w)
    N = size(X,2);
    if N==0,[y_bin,t] = deal([]);return;end
    edges = 1:bin_w:(N+bin_w);
    [~,~,loc]=histcounts(1:N,edges);
    y_bin = zeros(round(size(X).*[1,1/bin_w]));
    for ii = 1:size(X,1)
        y_bin(ii,:) = accumarray(loc(:),X(ii,:))...
            ./ accumarray(loc(:),1);
    end
    t = 0.5 * (edges(1:end-1) + edges(2:end));
end


function [M,mLine,traj,T,C] = plotTube(data,nvertex,thMerge,cm,ncurves,...
    TubeEdgeAlpha,TubeFaceAlpha,...
    meanColor,meanMarkerSize,meanLineWidth,...
    trajectoryAlpha,trajectoryWidth,trajectoryMarkerSize,trajectoryColor,...
    Thistitle,EventsTime,EventMarkerType,EvtLabel)

idx = 1:size(data,3);
% compute mean & std
m = median(data,3);
s = std(data,[],3);
zero = ceil(length(m)./2);
% select some trajectories close to mean
[~,i] = sort(arrayfun(@(id) max(sum(squeeze(data(:,:,id)) - m)), idx )); % finds trjaectories closest to mean
idx = idx(i);
allThalf = floor(numel(idx)/2);
ncurves = min(allThalf,ncurves);
idxTR = nan(1,ncurves);
while any(isnan(idxTR)) && ncurves < numel(idx)
    ridx = randperm(numel(idx),ncurves);   %randomly extract ncurves from the ones closest to the mean
    idxTR(isnan(idxTR)) = idx(ridx);
    idx(ridx) = [];
    idxTR(any((EventsTime(idxTR,:)+zero)<0 | (EventsTime(idxTR,:)+zero)>length(m),2)) = nan;
    ncurves = sum(isnan(idxTR));
end

%         [x,y,z]=tubeplot(m,s,nvertex,1e-9);
[x,y,z,m]=tuboplot(data,nvertex,thMerge);

% % link color to std
% val = arrayfun(@(idx)norm(s),1:length(s));
% val = val./max(val);
% val = val - min(val);
% val = [val(1) val val(end)];
% [~,~,bin]=histcounts(-val+1,linspace(0,1,256));
% CO = zeros([size(x),3]);
% CO(:,:,1) = ones(nvertex+1,1)*cm(bin,1)';
% CO(:,:,2) = ones(size(x,1),1)*cm(bin,2)';
% CO(:,:,3) = ones(size(x,1),1)*cm(bin,3)';

val = 1:size(data,2);

val = val./max(val);
val = val - min(val);
val = [val(1) val val(end)];
[~,~,bin]=histcounts(val,linspace(0,1,256));
% CO = zeros([size(x),3]);
CO(:,:,1) = ones(nvertex+1,1)*cm(bin,1)';
CO(:,:,2) = ones(size(x,1),1)*cm(bin,2)';
CO(:,:,3) = ones(size(x,1),1)*cm(bin,3)';

hold on
% plot the mash
M = mesh(x,y,z,CO,'EdgeAlpha',TubeEdgeAlpha,'FaceAlpha',TubeFaceAlpha);
% plot the mean
mLine = plot3(m(1,:),m(2,:),m(3,:),'Color',meanColor,'LineWidth',meanLineWidth);
mLine(2) = scatter3(m(1,zero),m(2,zero),m(3,zero),meanMarkerSize,"filled","o",...
    "MarkerFaceColor",ones(1,3),"MarkerEdgeColor",meanColor,"LineWidth",1);

% plot selected trajectories
traj = plot3(squeeze(data(1,:,idxTR)),squeeze(data(2,:,idxTR)),squeeze(data(3,:,idxTR)),...
    'Color',[trajectoryColor trajectoryAlpha],'LineWidth',trajectoryWidth);
s = scatter3(squeeze(data(1,zero,idxTR)),...
    squeeze(data(2,zero,idxTR)),...
    squeeze(data(3,zero,idxTR)),...
    trajectoryMarkerSize,trajectoryColor-.1,...
    "filled");
s.MarkerFaceAlpha = .6;
traj(end+1) = s;
% grid on;
set(gca,'XTick',[],'YTick',[],'ZTick',[]);
for ee = 1:size(EventsTime,2)
    thisEvt = EventsTime(:,ee);
    % mean markers
    mLine(2+ee) = scatter3(m(1,round(zero + mean(thisEvt))),...
        m(2,round(zero + mean(thisEvt))),...
        m(3,round(zero + mean(thisEvt))),...
        meanMarkerSize,...
        "filled",...
        EventMarkerType{ee},...
        "MarkerFaceColor",ones(1,3),"MarkerEdgeColor",meanColor,...
        "LineWidth",1);

    % traj markers
    for ii = idxTR(:)'
        s = scatter3(squeeze(data(1,zero+thisEvt(ii),ii)),...
            squeeze(data(2,zero+thisEvt(ii),ii)),...
            squeeze(data(3,zero+thisEvt(ii),ii)),...
            trajectoryMarkerSize,trajectoryColor-.1,...
            "filled",...
            EventMarkerType{ee});
        s.MarkerFaceAlpha = .6;
        traj(end+1) = s;
    end %ii
end %ee
T = title(Thistitle,'Interpreter','none');

colormap(cm);

ticks = arrayfun(@(x)x,linspace(1,size(data,2),11) - zero,...
    'UniformOutput',false);
tt = linspace(0,1,11);

C = colorbar('Ticks',tt,'TickLabels',ticks,'Limits',[0,1],'Box','off');
C.Label.String = 'Time [ms]';
C.Location = "North";
C.Ruler.TickLabelRotation = 0;

expAx = axes('Position',C.Position,'Color','none',...
    'XAxisLocation','top','YLim',[0 1],'XLim',[0 1]);
expAx.YAxis.Visible = false;
[tt,ii] = sort([mean(EventsTime,1)+zero zero]./length(m));
expAx.XTick = tt;
expAx.XTickLabel = EvtLabel(ii);
expAx.XTickLabelRotation = 30;
expAx.XAxis.TickDirection = 'none';

for ee = 1:size(EventsTime,2)
    thisEvt = EventsTime(:,ee);
    % mean markers
    line(expAx,(zero + mean(thisEvt))./length(m),.8,...
        'MarkerSize',meanMarkerSize/15,...
        'Marker',EventMarkerType{ee},...
        "MarkerFaceColor",ones(1,3),"MarkerEdgeColor",meanColor,...
        "LineWidth",1);
end
end

function cm = getR03cmap(bb,N)
if nargin<1
    bb = 'foo';
    N = 256;
elseif nargin<2
    N = 256;
end

if iscell(bb)
    cm = cell(size(bb));
    for b = 1:numel(bb)
        cm{b} = getR03cmap(bb{b});
    end
end

switch(bb)
    case {'prelesion1','preLesion1','postOp1A1','postop1a1',...
            'FailIntact_1','SuccessIntact_1'}
        %%
        %         N = 256;
        startHue = 2.5;
        rotations = 0;
        saturation = 1.5;
        gamma = 1.5;
        irange = [0.25 0.95];
        domain = [0.4 1];

        cm = (cubehelix(N+1,startHue,rotations,saturation,gamma,irange,domain));
        cm = cm(1:N,:);

    case {'prelesion2','preLesion2','postOp1A2','postop1a2',...
            'FailIntact_2','SuccessIntact_2'}
        %%
        %         N = 256;
        startHue = 2.75;
        rotations = 0;
        saturation = 1.5;
        gamma = 1.5;
        irange = [0.25 0.95];
        domain = [0.4 1];

        cm = (cubehelix(N+1,startHue,rotations,saturation,gamma,irange,domain));
        cm = cm(1:N,:);

    case {'prelesion3','preLesion3','postOp1A3','postop1a3',...
            'FailIntact_3','SuccessIntact_3'}
        %%
        %         N = 256;
        startHue = 3;
        rotations = 0;
        saturation = 1.5;
        gamma = 1.5;
        irange = [0.25 0.95];
        domain = [0.4 1];

        cm = (cubehelix(N+1,startHue,rotations,saturation,gamma,irange,domain));
        cm = cm(1:N,:);

    case {'postlesion1','postLesion1','postOp2A1','postop2a1',...
            'FailLesion_1','SuccessLesion_1'}
        %%
        %         N = 256;
        startHue = .9;
        rotations = 0;
        saturation = 2.1;
        gamma = 1.3;
        irange = [0.25 0.95];
        domain = [0.4 1];

        cm = (cubehelix(N+1,startHue,rotations,saturation,gamma,irange,domain));
        cm = cm(1:N,:);

    case {'postlesion2','postLesion2','postOp2A2','postop2a2',...
            'FailLesion_2','SuccessLesion_2'}
        %%
        %         N = 256;
        startHue = 1.15;
        rotations = 0;
        saturation = 2.1;
        gamma = 1.3;
        irange = [0.25 0.95];
        domain = [0.4 1];

        cm = (cubehelix(N+1,startHue,rotations,saturation,gamma,irange,domain));
        cm = cm(1:N,:);

    otherwise
        block = {'preLesion1','preLesion2','preLesion3','postLesion1','postLesion2'};
        figure('Color',[1 1 1]);
        cm = cell(size(block));
        for bb = 1:numel(block)
            ax = subplot(1,numel(block),bb);
            cm{bb} = getR03cmap(block{bb},N);
            cmAx = image('Parent',ax, 'CData',permute(cm{bb},[1,3,2]));
            ylim([0 N]+.5);
            title(block{bb});
            ax.XAxis.Visible = false;
            ticks = 1:N;
            ax.YTick = ticks(1:round(N/10):N);
        end
end
end