function i_storeM(obj, data, type)
%I_STOREM  Store a diagnostic result in the M_ metrics struct array.
%
%   I_STOREM(OBJ, DATA, TYPE) wraps DATA in the standard M_ struct
%   template (fields: type, date, condition, data, Area) and updates
%   OBJ.M_ using the same replace-or-append logic as COMPUTEMETRICS:
%
%     * If M_ is still the empty initial placeholder, it is replaced.
%     * If OBJ.appendM == true, the new entry is always appended.
%     * Otherwise an existing entry with the same type / condition / Area
%       is overwritten; if none exists, the new entry is appended.
%
%   This method is private and is called by the diagnostic class methods
%   (selectDimension, crossValReconstruct, crossValDecode, alignSessions).
%
%   See also NeuralEmbedding.computeMetrics, NeuralEmbedding.initMstruct

Mstr = obj.initMstruct(data, type);

if isempty(obj.M)
    % M_ is still the empty initialisation placeholder
    obj.M_ = Mstr;
elseif obj.appendM
    obj.M_ = [obj.M_ Mstr];
else
    % Replace the most recent entry with the same type / condition / area
    idx = ([obj.M_.type] == Mstr.type) & ...
          ([obj.M_.condition] == Mstr.condition) & ...
          ([obj.M_.Area] == Mstr.Area);
    if any(idx)
        obj.M_(idx) = Mstr;
    else
        obj.M_ = [obj.M_ Mstr];
    end
end
end
