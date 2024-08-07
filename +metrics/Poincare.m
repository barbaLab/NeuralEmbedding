mixedConds = {'Intact_1','Intact_2','Lesion_1','Lesion_2'};
nComp = 6;
nBlk = numel(mixedConds);
an = cell(12,numel(mixedConds),3);
theta = nan(12,numel(mixedConds),3);
conds = an;

for area = 1:3
    for aa = 1:12
        if isempty(AllProjected{aa}{area})
            [an{aa,:,area}] = deal(aa);
            conds(aa,:,area) = mixedConds;
            continue;
        end
        for c = 1:numel(mixedConds)
            cc = mixedConds{c};
            idx = contains({AllProjected{aa}{area}.condition},cc);
            dat = cat(3,AllProjected{aa}{area}(idx).data);
            if not(isempty(dat))
                dat = dat(1:nComp,:,:);
            end
            theta(aa,c,area) = pcre(dat,200);
            an{aa,c,area} = aa;
            conds{aa,c,area} = cc;


            if ~isempty(dat)
                [~,~,R(aa,c,area),~] = circfit(mean(dat(1,:),3),mean(dat(2,:),3));
            end
        end
    end
end

%%
area = cat(3,repmat({'RFA'},12,numel(mixedConds)),...
    repmat({'S1'},12,numel(mixedConds)),...
    repmat({'All'},12,numel(mixedConds)));
an = cellfun(@num2str,an,'UniformOutput',false);
figure;
g = gramm('x',theta,'column',conds,'row',an,'fig',area);
g.stat_bin('geom','overlaid_bar','normalization','probability')
g.draw


%%

[success,fail,noAtt,nAn,nBlck] = getSuccessRate(tankObj);
succ = success./(success+fail);

Deficit = false(12,4,3);
for a = 1:12
    for block = 1:5
        deltaScore = max(succ(a,1:3)) - min(succ(a,4:5));
        if deltaScore > .3
            Deficit(a,3:4,:) = true;

            an = (1:12)'*ones(1,4);
            an = repmat(an,1,1,3);
            area = cat(3,repmat({'RFA'},12,4),...
                repmat({'S1'},12,4),...
                repmat({'All'},12,4));
            % disp = cellfun(@(x)std(x,'omitnan'),theta);
        end
    end
end
bl = repmat(ones(12,1)*[1:4],1,1,3);

succ(3:8,1:2) =  succ(3:8,2:3);
succ(:,3) = [];
succ = repmat(succ,1,1,3);


figure
g = gramm('x',theta(:),'y',succ(:),'color',an(:),'column',area(:));
g.set_names('x','Trajectory tangling','y','Success Rate','color','Animal');
g.set_point_options('markers',{'o','^'},'base_size',7);
% g.geom_abline('intercept',1);
g.stat_glm('geom','line','disp_fit',false,'fullrange',false);
g.set_line_options('styles',{'--'},'base_size',.8);
g.geom_point('alpha',.8);
g.draw;



figure
theta_M = [mean(theta(:,1:2,:),2,'omitmissing') mean(theta(:,3:4,:),2,'omitmissing')];
bl_M = [mean(bl(:,1:2,:),2) mean(bl(:,3:4,:),2)];
g = gramm('y',theta(:),'x',bl(:),'color',an(:),'column',area(:));
g.set_names('x','Days','y','Trajectory tangling','color','Animal');
% g.set_point_options('markers',{'o','^'},'base_size',7);
% g.geom_abline('intercept',1);
% g.stat_glm('geom','line','disp_fit',false,'fullrange',false);
% g.set_line_options('styles',{'--'},'base_size',.8);
g.geom_point('alpha',.8);
g.draw()
g.update('x',bl_M(:),'y',theta_M(:));
g.geom_line()


g.draw;

%%
T = ones(12,1)*[1:nBlk];
T = repmat(T,1,1,3);

an = (1:12)'*ones(1,nBlk);
an = repmat(an,1,1,3);

area = cat(3,repmat({'RFA'},12,nBlk),...
    repmat({'S1'},12,nBlk),...
    repmat({'All'},12,nBlk));
% disp = cellfun(@std,theta);

figure("Position",[102.6000 339.4000 668 328.8000]);
th = theta(:);
T = T(:);
area = area(:);
an = an(:);
g= gramm('y', th(~isnan(th) ),'x',T(~isnan(th) ),...
'color',an(~isnan(th)),'column',area(~isnan(th)));
g.geom_jitter();
g.set_names('x','Day','y','theta dispersion','color','Rats');
g.axe_property('XTick',[1:4],'XTickLabels',["Pre1","Pre2","Post1","Post2"],...
    'xlim',[0.5 5.5]);

g.geom_polygon('x',{[2.5 4.5 4.5 2.5]},'y',{[-1 -1 1 1]})
g.set_color_options('map','brewer3')

g.stat_glm('geom','line','disp_fit',false,'fullrange',false);
g.set_line_options("base_size",2.5)
g.draw();

Deficit_ = Deficit;
Deficit_(:,3:4,:) = 1;
%% Aggregate on deficit
figure
area = cat(3,repmat({'RFA'},12,4),repmat({'S1'},12,4),repmat({'All'},12,4));
g = gramm('x',Deficit(:),'y',theta(:),'color',Deficit(:),'column',area(:));
g.stat_boxplot('width',1,'notch',true);
g.geom_jitter()
g.draw;

figure
g = gramm('x',Deficit_(:),'y',theta(:),'color',Deficit_(:));
g.stat_boxplot('width',1,'notch',true);
g.geom_jitter()
g.draw;

%% Write everything to table
hasDeficit = (max(succ(:,1:3),[],2)-min(succ(:,4:5),[],2) ) >.3;
hasDeficit(6) = true;
t = table(th(:),mixedConds(T(:))',an(:),area(:),succ(:),Deficit(:),Deficit_(:),hasDeficit(an(:)),'VariableNames',{'tangl','block','animal','area','score','deficit','deficitB','isLesioned'});
writetable(t,'C:\Users\Fede\Documents\MATLAB\R03 figures\Results\Tangl\Tangl.GPFA.csv')
%% New tanglind metric
function [theta ] = pcre(data,idx)
if isempty(data)
    theta = nan;
    return;
end

m = mean(data,3);



pt = m(:,idx);
pt_1 = m(:,idx+1);

n = pt_1 - pt;
n = n./vecnorm(n);
p = null(n);

X = reshape(data,size(data,1),[],1);


end


