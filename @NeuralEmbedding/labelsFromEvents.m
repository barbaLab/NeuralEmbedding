function y = labelsFromEvents(obj, eventNames)
%LABELSFROMEVENTS Build a per-time-bin label vector from stored events.
%
%   Y = LABELSFROMEVENTS(OBJ, EVENTNAMES) creates a label vector Y with
%   one element per time-bin (matching the trial-concatenated data returned
%   by methods such as crossValDecode and crossValReconstruct).
%
%   Each time bin in each trial is labelled with the name of the *most
%   recent* event (from EVENTNAMES) that has occurred up to that bin,
%   or '0' if no requested event has occurred yet.
%
%   The method respects the current condition mask (cMask) of the object:
%   only masked trials are included, in the same order as OBJ.S and OBJ.E.
%
%   Inputs
%   ------
%   obj        : scalar NeuralEmbedding object (must have events stored
%                via addEvents).
%   eventNames : string, char, string array, or cellstr — event name(s) to
%                use as label sources.  The order determines priority: if
%                two events occur at the same timestamp, the first in
%                EVENTNAMES wins.
%                Pass "all" to use every unique event name present in the
%                object (names used as labels, in order of first occurrence).
%
%   Output
%   ------
%   y : T x 1 categorical vector (T = total time-bins across all *masked*
%       trials, in the same order as OBJ.S(:,1)).
%       Categories are the requested event names plus '0' (= no event yet).
%
%   Notes
%   -----
%   * Only the masked trials (cMask) are included; the result aligns with
%     the output of OBJ.S, OBJ.E, and the CV diagnostic methods.
%   * Events that fall outside the trial time window are ignored.
%   * If a trial has no events in EVENTNAMES, all bins in that trial
%     are labelled '0'.
%
%   Example
%   -------
%   % Label time bins as 'Cue', 'Go', or '0' (pre-event)
%   y = NE.labelsFromEvents({'Cue','Go'});
%   res = NE.crossValDecode(y, 5);
%
%   % Use all events
%   y = NE.labelsFromEvents("all");
%
%   See also NeuralEmbedding.addEvents, NeuralEmbedding.crossValDecode

% --- Validate inputs ---
if ~isscalar(obj)
    error('NeuralEmbedding:labelsFromEvents:notScalar', ...
        'OBJ must be a scalar NeuralEmbedding object. Loop over sessions in caller.');
end
if isempty(obj.Events)
    error('NeuralEmbedding:labelsFromEvents:noEvents', ...
        'No events stored. Call addEvents first.');
end

eventNames = string(eventNames);

% Resolve "all" → all unique event names in order of first appearance
if isscalar(eventNames) && eventNames == "all"
    evtNames_ = string({obj.Events_.Name});
    [~, firstIdx] = unique(evtNames_, 'stable');
    eventNames = evtNames_(sort(firstIdx));
end

% --- Get masked trial indices (1-based into full trial list) ---
maskedTrialIdx = find(obj.cMask(:)');   % 1 x nMasked

% --- Build trial-level lookup tables ---
% S respects cMask; use first area column for time-bin counts
S = obj.S;
S = S(:, 1);  % first area column (nMasked x 1)
nMasked       = numel(S);
nBinsPerTrial = cellfun(@(s) size(s, 2), S);
T_total       = sum(nBinsPerTrial);

% Time vectors filtered by cMask (and tMask, subsampling)
trialTimes = obj.TrialTime;  % cell array, length = nMasked

% Pre-allocate label vector
yStr = repmat("0", T_total, 1);

% Pull event fields once
evtTrialIdx = [obj.Events_.Trial];
evtTs       = [obj.Events_.Ts];
evtNameArr  = string({obj.Events_.Name});

% Filter to requested event names only
isRequested  = ismember(evtNameArr, eventNames);
[~, evtPriority] = ismember(evtNameArr, eventNames);  % 0 if not requested

offset = 0;
for k = 1:nMasked
    origTr = maskedTrialIdx(k);   % original (unmasked) trial index
    tVec   = trialTimes{k};       % 1 x nBins or nBins x 1 (masked+subsampled)
    nBins  = nBinsPerTrial(k);
    if isempty(tVec) || nBins == 0
        offset = offset + nBins;
        continue;
    end
    tVec = tVec(:)';  % 1 x nBins row

    % Events for this (original) trial
    inTrial = (evtTrialIdx == origTr) & isRequested;
    if ~any(inTrial)
        offset = offset + nBins;
        continue;
    end

    ts_tr    = evtTs(inTrial);
    names_tr = evtNameArr(inTrial);
    prio_tr  = evtPriority(inTrial);

    % Sort by timestamp, then by priority (lower index = higher priority)
    [ts_sorted, sortOrd] = sort(ts_tr);
    names_sorted = names_tr(sortOrd);
    prio_sorted  = prio_tr(sortOrd);

    % For ties in timestamp: keep the event with the lowest eventNames index
    uniqueTs = unique(ts_sorted);
    for uIdx = 1:numel(uniqueTs)
        same = (ts_sorted == uniqueTs(uIdx));
        if sum(same) > 1
            [~, bestRel] = min(prio_sorted(same));
            keep = find(same);
            remove = keep;  remove(bestRel) = [];
            names_sorted(remove) = [];
            prio_sorted(remove)  = [];
            ts_sorted(remove)    = [];
        end
    end

    % Assign labels: each bin gets the name of the most recent event before it
    binLabels = repmat("0", nBins, 1);
    for evIdx = 1:numel(ts_sorted)
        afterTs = tVec >= ts_sorted(evIdx);
        binLabels(afterTs) = names_sorted(evIdx);
    end

    yStr(offset + (1:nBins)) = binLabels;
    offset = offset + nBins;
end

% Convert to categorical with a consistent category set
allCats = ["0", eventNames(:)'];
y = categorical(yStr, allCats);
end
