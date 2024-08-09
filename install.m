rootFolder = pwd;
CCAfolder   = fullfile(rootFolder,"+embedding/","+CCA/+mcca");
UMAPfolder  = fullfile(rootFolder,"+embedding/","+UMAP/umapAndEpp");
tSNEfolder  = fullfile(rootFolder,"+embedding/","+tSNE/");

if ~existsnotempty(CCAfolder)
    msetCCAgit = "https://github.com/lcparra/mcca.git";
    gitclone(msetCCAgit,CCAfolder);
end

if ~existsnotempty(UMAPfolder)
    UMAProot = fullfile(rootFolder,"+embedding/","+UMAP");
    zipfile = UMAPfolder+".zip";
    unzip(zipfile,UMAPfolder);
    addpath(genpath(UMAPfolder))
end


function value = existsnotempty(path)
   value = logical(exist(path,"dir"));
   value = value && (numel(dir(path)) > 2);
end