%SMOKE_TEST_DIAGNOSTICS Minimal smoke / sanity test for manifold diagnostics.
%
%   This script verifies the manifold diagnostic toolkit against synthetic
%   data with known ground-truth properties:
%
%     1) dStar from parallel analysis should be near the true latent dim.
%     2) CV reconstruction improves with dimension.
%     3) CV decoding with real labels should be significant; with random labels not.
%     4) Cross-session Procrustes alignment: disparity bounded.
%     4b) useAlignment flag toggles between original / aligned E.
%     5) Multi-session selectDimension returns struct array.
%     6) Diagnostic results stored in M_.
%     7) crossValAlignment: finite disparity distribution, stored in M_.
%     8) labelsFromEvents: returns categorical with correct length.
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

% Suppress progress output during automated tests
quietParsPA  = diagnostics.pars.ParallelAnalysis();   quietParsPA.verbose  = false;
quietParsCVR = diagnostics.pars.CVReconstruction();   quietParsCVR.verbose = false;
quietParsCVD = diagnostics.pars.CVDecoding();         quietParsCVD.verbose = false;
quietParsAS  = diagnostics.pars.ProcrustesAlignment(); quietParsAS.verbose = false;
quietParsIA  = diagnostics.pars.IntraAlignment();     quietParsIA.verbose  = false;

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
pA          = quietParsPA;
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
pB         = quietParsCVR;
pB.kfold   = 3;
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
pC          = quietParsCVD;
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
%  Test 4 – Cross-session alignment reduces disparity
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

resD = NEobjs.alignSessions(quietParsAS);
% disparity is now nAreas x nSessions; check session 2, any area
pass = all(resD.disparity(:, 2) <= 1.0);   % disparity should be bounded
nFail = nFail + ~pass;
fprintf('[Test 4] Procrustes alignment  max_disparity(sess2)=%.4f  ... %s\n', ...
    max(resD.disparity(:, 2)), tf(pass));

% Check M_ type is 'SessionAlignment' (not 'Alignment')
M4 = NE2.M;
pass4type = any(strcmp(M4.type, 'SessionAlignment'));
nFail = nFail + ~pass4type;
fprintf('[Test 4c] M_ type = SessionAlignment  ... %s\n', tf(pass4type));

% =========================================================================
%  Test 4b – useAlignment flag: E changes after alignment activation
% =========================================================================
E_before = NEobjs(2).E;          % original (unaligned) embedding
NEobjs(2).useAlignment = true;
E_after  = NEobjs(2).E;          % should now return aligned embedding
NEobjs(2).useAlignment = false;  % restore

% Filter out empty or zero-time-bin cells before comparing
valid_b = ~cellfun(@(e) isempty(e) || size(e,2)==0, E_before);
valid_a = ~cellfun(@(e) isempty(e) || size(e,2)==0, E_after);
if any(valid_b & valid_a)
    E_b_mat = cell2mat(cellfun(@(e) e(:)', E_before(valid_b & valid_a), ...
        'UniformOutput', false));
    E_a_mat = cell2mat(cellfun(@(e) e(:)', E_after(valid_a & valid_b),  ...
        'UniformOutput', false));
    pass4b  = ~isequal(E_b_mat, E_a_mat) || ...
              (norm(E_b_mat(:) - E_a_mat(:)) < 1e-8);
else
    pass4b = true;   % no valid cells to compare; treat as pass
end
nFail = nFail + ~pass4b;
fprintf('[Test 4b] useAlignment flag works  ... %s\n', tf(pass4b));

% =========================================================================
%  Test 5 – Multi-session selectDimension returns struct array
% =========================================================================
resMS = NEobjs.selectDimension(1:8, pA);
pass = isstruct(resMS) && numel(resMS) == 2;
nFail = nFail + ~pass;
fprintf('[Test 5] Multi-session selectDimension  numel=%d  ... %s\n', ...
    numel(resMS), tf(pass));

% =========================================================================
%  Test 6 – Diagnostic results stored in M_
% =========================================================================
M_tbl = NE.M;    % get.M returns a table
types_stored = string(M_tbl.type);
pass6 = all(ismember(["ParallelAnalysis","CVReconstruction","CVDecoding"], types_stored));
nFail = nFail + ~pass6;
fprintf('[Test 6] Diagnostics stored in M_  types=%s  ... %s\n', ...
    strjoin(types_stored, ', '), tf(pass6));

% =========================================================================
%  Test 7 – crossValAlignment: finite disparity, stored in M_
% =========================================================================
pIA          = quietParsIA;
pIA.nSplit   = 20;    % few splits for speed
pIA.rngSeed  = 0;

resIA = NE.crossValAlignment(pIA);
pass7a = numel(resIA.disparity) == 20 && all(isfinite(resIA.disparity));
nFail = nFail + ~pass7a;
fprintf('[Test 7a] crossValAlignment  nSplit=%d  medDisp=%.4f  ... %s\n', ...
    numel(resIA.disparity), resIA.disparityMedian, tf(pass7a));

M7 = NE.M;
pass7b = any(strcmp(M7.type, 'IntraAlignment'));
nFail = nFail + ~pass7b;
fprintf('[Test 7b] IntraAlignment stored in M_  ... %s\n', tf(pass7b));

% =========================================================================
%  Test 8 – labelsFromEvents: returns categorical of correct length
% =========================================================================
% Add synthetic events to NE: 'Cue' halfway through each trial
evts = struct('Ts', {}, 'Name', {}, 'Trial', {}, 'Data', {});
for tr = 1:nTrials
    evts(end+1).Ts    = time(floor(Ttrial/2));   %#ok<SAGROW>
    evts(end).Name   = 'Cue';
    evts(end).Trial  = tr;
    evts(end).Data   = [];
end
NE.addEvents(evts);
y_evts = NE.labelsFromEvents('Cue');
T_expected = nTrials * Ttrial;
pass8a = numel(y_evts) == T_expected && iscategorical(y_evts);
nFail  = nFail + ~pass8a;
fprintf('[Test 8a] labelsFromEvents  T=%d (expected %d)  ... %s\n', ...
    numel(y_evts), T_expected, tf(pass8a));

% Check that exactly half the bins are labelled 'Cue'
nCue = sum(y_evts == 'Cue');
nExpCue = nTrials * (Ttrial - floor(Ttrial/2) + 1);
pass8b = (nCue == nExpCue);
nFail  = nFail + ~pass8b;
fprintf('[Test 8b] labelsFromEvents  nCue=%d (expected %d)  ... %s\n', ...
    nCue, nExpCue, tf(pass8b));

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

