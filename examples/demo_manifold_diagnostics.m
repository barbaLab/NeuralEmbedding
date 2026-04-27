%DEMO_MANIFOLD_DIAGNOSTICS End-to-end demo of the manifold diagnostic toolkit.
%
%   This script demonstrates all four diagnostic capabilities using
%   synthetic low-dimensional latent data:
%
%     A) Dimension selection via parallel analysis (selectDimension)
%     B) Cross-validated reconstruction (crossValReconstruct)
%     C) Cross-validated decoding + permutation test (crossValDecode)
%     D) Multi-session alignment (alignSessions)
%
%   No real data are required: synthetic sessions are generated in-script.
%   All results are printed to the console; figures are saved to the
%   current directory as PNG files.
%
%   Usage
%   -----
%   Run this script from the NeuralEmbedding root directory after
%   adding it to the MATLAB path:
%       addpath(genpath('path/to/NeuralEmbedding'));
%       run examples/demo_manifold_diagnostics.m
%
%   See also NeuralEmbedding, selectDimension, crossValReconstruct,
%            crossValDecode, alignSessions

clear; clc;

rng(42, 'twister');

fprintf('=============================================================\n');
fprintf('  NeuralEmbedding – Manifold Diagnostics Demo\n');
fprintf('=============================================================\n\n');

% =========================================================================
%  Synthetic data parameters
% =========================================================================
trueD  = 3;     % true latent dimensionality
nUnits = 40;    % simulated neurons
nTrials = 60;   % trials per session
T_trial = 50;   % time bins per trial
nSessions = 2;  % sessions to simulate
noiseStd = 0.5; % additive noise level
fs = 1000;      % sampling rate (Hz) – for NeuralEmbedding constructor
dt = 1/fs;      % 1 ms bins
time = (0 : T_trial-1) * dt;

% =========================================================================
%  Helper: generate one synthetic session
% =========================================================================
function [NE, y_trial] = make_session(trueD, nUnits, nTrials, T_trial, ...
        time, fs, noiseStd, sessionLabel, seed)
    rng(seed, 'twister');

    % Random mixing matrix (nUnits x trueD)
    C = randn(nUnits, trueD);

    % Random trial labels (2 classes)
    y_trial = randi([1 2], nTrials, 1);

    % Build data: nUnits x T x nTrials
    data = zeros(nUnits, T_trial, nTrials);
    for tr = 1:nTrials
        Z = randn(trueD, T_trial);                % latent trajectory
        Z(1, :) = Z(1, :) + 2 * (y_trial(tr)-1); % label-dependent offset
        data(:, :, tr) = C * Z + noiseStd * randn(nUnits, T_trial);
    end

    % Create NeuralEmbedding object
    NE = NeuralEmbedding(data, ...
        'time', time, ...
        'fs',   fs, ...
        'condition', string(y_trial), ...
        'session', sessionLabel, ...
        'animal',  'SyntheticAnimal');
end

% =========================================================================
%  Create two synthetic sessions
% =========================================================================
fprintf('Generating synthetic data (%d sessions, %d neurons, %d trials)...\n\n', ...
    nSessions, nUnits, nTrials);

[NE1, y1] = make_session(trueD, nUnits, nTrials, T_trial, time, fs, noiseStd, 'Sess1', 1);
[NE2, y2] = make_session(trueD, nUnits, nTrials, T_trial, time, fs, noiseStd, 'Sess2', 2);

NEobjs = [NE1, NE2];

% =========================================================================
%  A) Dimension selection via parallel analysis
% =========================================================================
fprintf('----- A) Dimension selection (parallel analysis) ------------\n');
pA          = diagnostics.pars.ParallelAnalysis();
pA.nShuffle = 100;   % fewer for demo speed
pA.alpha    = 0.05;
pA.rngSeed  = 0;

resA = NEobjs.selectDimension(1:12, pA);

for ss = 1:nSessions
    fprintf('  %s.%s:  dStar = %d  (true dim = %d)\n', ...
        resA(ss).animal, resA(ss).session, resA(ss).dStar, trueD);
end

% Plot eigenvalues for first session
fig = figure('Visible','off');
bar(resA(1).dimsTested, resA(1).eigReal, 'FaceColor', [.3 .6 .9]); hold on;
plot(resA(1).dimsTested, resA(1).nullQuantile, 'r--', 'LineWidth', 1.5);
xlabel('Component'); ylabel('Eigenvalue');
title(sprintf('%s.%s – Parallel Analysis (dStar=%d)', ...
    resA(1).animal, resA(1).session, resA(1).dStar));
legend('Real eigenvalue', sprintf('Null %.0f%% quantile', (1-pA.alpha)*100));
saveas(fig, 'demo_A_parallel_analysis.png');
fprintf('  Saved: demo_A_parallel_analysis.png\n\n');

% =========================================================================
%  B) Cross-validated reconstruction
% =========================================================================
fprintf('----- B) Cross-validated reconstruction --------------------\n');
dimToTest = 1:8;
pB        = diagnostics.pars.CVReconstruction();
pB.kfold  = 5;
pB.rngSeed = 0;

reconCorr = zeros(nSessions, numel(dimToTest));
for ss = 1:nSessions
    for dd = 1:numel(dimToTest)
        r = NEobjs(ss).crossValReconstruct(dimToTest(dd), pB);
        reconCorr(ss, dd) = r.reconCorrMean;
    end
end

fig = figure('Visible','off');
plot(dimToTest, reconCorr(1,:), 'b-o', dimToTest, reconCorr(2,:), 'r-s');
xline(trueD, 'k--', 'True dim');
xlabel('Latent dimension'); ylabel('CV Pearson r (reconstruction)');
title('Cross-validated reconstruction vs. dimensionality');
legend('Session 1', 'Session 2', 'True dim');
saveas(fig, 'demo_B_cv_reconstruction.png');
fprintf('  Session 1 r at d=%d: %.3f\n', trueD, reconCorr(1, trueD));
fprintf('  Session 2 r at d=%d: %.3f\n', trueD, reconCorr(2, trueD));
fprintf('  Saved: demo_B_cv_reconstruction.png\n\n');

% =========================================================================
%  C) Cross-validated decoding + permutation test
% =========================================================================
fprintf('----- C) CV decoding + permutation test --------------------\n');
pC            = diagnostics.pars.CVDecoding();
pC.kfold      = 5;
pC.nPerm      = 200;   % fewer for demo speed
pC.rngSeed    = 0;
pC.permMode   = 'global';

% Real labels – should decode significantly
resC_real = NE1.crossValDecode(y1, trueD, pC);
fprintf('  [Real labels]  Acc=%.1f%%  BalAcc=%.1f%%  p=%.4f  z=%.2f\n', ...
    resC_real.accMean*100, resC_real.balAccMean*100, ...
    resC_real.pValue, resC_real.effectZ);

% Random labels – should not decode significantly
y_rand     = randi([1 2], nTrials, 1);
resC_rand  = NE1.crossValDecode(y_rand, trueD, pC);
fprintf('  [Random labels] Acc=%.1f%%  BalAcc=%.1f%%  p=%.4f  z=%.2f\n', ...
    resC_rand.accMean*100, resC_rand.balAccMean*100, ...
    resC_rand.pValue, resC_rand.effectZ);

% Plot null distributions
fig = figure('Visible','off');
subplot(1,2,1);
histogram(resC_real.statNull * 100, 20, 'FaceColor', [.8 .8 .8]); hold on;
xline(resC_real.accMean * 100, 'r', 'LineWidth', 2);
xlabel('Null accuracy (%)'); title(sprintf('Real labels  p=%.3f', resC_real.pValue));
subplot(1,2,2);
histogram(resC_rand.statNull * 100, 20, 'FaceColor', [.8 .8 .8]); hold on;
xline(resC_rand.accMean * 100, 'b', 'LineWidth', 2);
xlabel('Null accuracy (%)'); title(sprintf('Random labels  p=%.3f', resC_rand.pValue));
sgtitle('CV Decoding – Permutation Test');
saveas(fig, 'demo_C_cv_decoding.png');
fprintf('  Saved: demo_C_cv_decoding.png\n\n');

% =========================================================================
%  D) Multi-session alignment (Procrustes)
% =========================================================================
fprintf('----- D) Multi-session alignment (Procrustes) --------------\n');

% Compute embeddings first
NEobjs(1).numPC = trueD;
NEobjs(2).numPC = trueD;
NEobjs(1).findEmbedding('PCA');
NEobjs(2).findEmbedding('PCA');

pD            = diagnostics.pars.ProcrustesAlignment();
pD.refSession = 1;
pD.allowScale = false;

resD = NEobjs.alignSessions(pD);
fprintf('  Session 1 (ref):  disparity = %.4f\n', resD.disparity(1));
fprintf('  Session 2:        disparity = %.4f  mean angle = %.2f°  distCorr = %.3f\n', ...
    resD.disparity(2), rad2deg(resD.meanPrincipalAngle(2)), resD.distCorr(2));

% Show pre- vs post-alignment principal angles
fig = figure('Visible','off');
bar(rad2deg(resD.principalAngles{2}));
xlabel('Principal angle index'); ylabel('Angle (degrees)');
title('Session 2 → Session 1: principal angles after alignment');
saveas(fig, 'demo_D_alignment.png');
fprintf('  Saved: demo_D_alignment.png\n\n');

% =========================================================================
fprintf('=============================================================\n');
fprintf('  Demo complete.\n');
fprintf('  PNG figures saved to: %s\n', pwd);
fprintf('=============================================================\n');
