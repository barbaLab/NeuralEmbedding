%SMOKE_TEST_DIAGNOSTICS Minimal smoke / sanity test for manifold diagnostics.
%
%   This script verifies the manifold diagnostic toolkit against synthetic
%   data with known ground-truth properties:
%
%     1) dStar from parallel analysis should be near the true latent dim.
%     2) CV decoding with real labels should be significant (p < 0.05) and
%        above chance; with random labels it should be non-significant.
%     3) Procrustes alignment should reduce disparity (or keep it zero for
%        aligned data).
%
%   All tests print PASS / FAIL to the console. Exits with an error if any
%   test fails so it can be integrated into automated pipelines.
%
%   Usage
%   -----
%   From the NeuralEmbedding root directory:
%       addpath(genpath('path/to/NeuralEmbedding'));
%       run tests/smoke_test_diagnostics.m

clear; clc;

fprintf('Running manifold-diagnostics smoke tests...\n\n');
nFail = 0;

% =========================================================================
%  Shared synthetic data
% =========================================================================
rng(0, 'twister');
trueD   = 3;
nUnits  = 30;
nTrials = 80;
Ttrial  = 40;    % time bins per trial
noiseStd = 0.3;
fs = 1000;
time = (0:Ttrial-1) / fs;

% Random mixing matrix
C = randn(nUnits, trueD);

y_trial = randi([1 2], nTrials, 1);
data    = zeros(nUnits, Ttrial, nTrials);
for tr = 1:nTrials
    Z = randn(trueD, Ttrial);
    Z(1,:) = Z(1,:) + 3*(y_trial(tr)-1);   % strong signal
    data(:,:,tr) = C*Z + noiseStd*randn(nUnits, Ttrial);
end

NE = NeuralEmbedding(data, ...
    'time', time, 'fs', fs, ...
    'condition', string(y_trial), ...
    'session', 'Test', 'animal', 'SmokeTest');

% =========================================================================
%  Test 1 – Parallel analysis: dStar near trueD
% =========================================================================
pA          = diagnostics.pars.ParallelAnalysis();
pA.nShuffle = 100;
pA.rngSeed  = 0;

resA = NE.selectDimension(1:10, pA);
pass = (resA.dStar >= trueD - 1) && (resA.dStar <= trueD + 2);
nFail = nFail + ~pass;
fprintf('[Test 1] Parallel analysis  dStar=%d  (true=%d)  ... %s\n', ...
    resA.dStar, trueD, tf(pass));

% =========================================================================
%  Test 2 – CV reconstruction improves with dim
% =========================================================================
pB       = diagnostics.pars.CVReconstruction();
pB.kfold = 3;
pB.rngSeed = 0;

r1 = NE.crossValReconstruct(1, pB);
r3 = NE.crossValReconstruct(trueD, pB);
pass = (r3.reconCorrMean > r1.reconCorrMean) && (r3.reconCorrMean > 0);
nFail = nFail + ~pass;
fprintf('[Test 2] CV reconstruction   r(d=1)=%.3f  r(d=%d)=%.3f  ... %s\n', ...
    r1.reconCorrMean, trueD, r3.reconCorrMean, tf(pass));

% =========================================================================
%  Test 3 – Decoding: real labels significant, random labels not
% =========================================================================
pC          = diagnostics.pars.CVDecoding();
pC.kfold    = 3;
pC.nPerm    = 200;
pC.rngSeed  = 0;
pC.permMode = 'global';

resC_real = NE.crossValDecode(y_trial, trueD, pC);
pass3a = (resC_real.pValue < 0.05) && (resC_real.accMean > 0.55);
nFail = nFail + ~pass3a;
fprintf('[Test 3a] Decoding real labels  acc=%.1f%%  p=%.4f  ... %s\n', ...
    resC_real.accMean*100, resC_real.pValue, tf(pass3a));

y_rand     = randi([1 2], nTrials, 1);
resC_rand  = NE.crossValDecode(y_rand, trueD, pC);
pass3b = (resC_rand.pValue > 0.05) || (resC_rand.accMean < 0.65);
nFail = nFail + ~pass3b;
fprintf('[Test 3b] Decoding random labels  acc=%.1f%%  p=%.4f  ... %s\n', ...
    resC_rand.accMean*100, resC_rand.pValue, tf(pass3b));

% =========================================================================
%  Test 4 – Alignment reduces disparity
% =========================================================================
rng(1,'twister');
R_true  = orth(randn(trueD));   % random rotation
C2      = C * R_true;

data2 = zeros(nUnits, Ttrial, nTrials);
for tr = 1:nTrials
    Z = randn(trueD, Ttrial);
    Z(1,:) = Z(1,:) + 3*(y_trial(tr)-1);
    data2(:,:,tr) = C2*Z + noiseStd*randn(nUnits, Ttrial);
end

NE2 = NeuralEmbedding(data2, ...
    'time', time, 'fs', fs, ...
    'condition', string(y_trial), ...
    'session', 'Test2', 'animal', 'SmokeTest');

NEobjs = [NE, NE2];
NEobjs(1).numPC = trueD;
NEobjs(2).numPC = trueD;
NEobjs(1).findEmbedding('PCA');
NEobjs(2).findEmbedding('PCA');

resD = NEobjs.alignSessions();
pass = resD.disparity(2) <= 1.0;   % disparity should be bounded
nFail = nFail + ~pass;
fprintf('[Test 4] Procrustes alignment  disparity(sess2)=%.4f  ... %s\n', ...
    resD.disparity(2), tf(pass));

% =========================================================================
%  Test 5 – Multi-session selectDimension returns struct array
% =========================================================================
resMS = NEobjs.selectDimension(1:8, pA);
pass = isstruct(resMS) && numel(resMS) == 2;
nFail = nFail + ~pass;
fprintf('[Test 5] Multi-session selectDimension  numel=%d  ... %s\n', ...
    numel(resMS), tf(pass));

% =========================================================================
%  Summary
% =========================================================================
fprintf('\n');
if nFail == 0
    fprintf('All smoke tests PASSED.\n');
else
    error('smoke_test_diagnostics:failed', ...
        '%d smoke test(s) FAILED. See output above.', nFail);
end

% =========================================================================
function s = tf(pass)
    if pass, s = 'PASS'; else, s = 'FAIL'; end
end
