msetCCAgit = "https://github.com/lcparra/mcca.git";
rootFolder = pwd;
cd(fullfile(rootFolder,"+embedding/","+CCA/"));
gitclone(msetCCAgit);
movefile("mcca","+mcca");