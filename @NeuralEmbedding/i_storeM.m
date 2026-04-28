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

type = string(type);     % ensure string for reliable == comparison
Mstr = obj.initMstruct(data, type);

% Use M_ directly (not the public getter) to avoid getter transformation
if all(arrayfun(@(t) all(structfun(@isempty,t)), obj.M_))
    % M_ is still the empty initialisation placeholder
    obj.M_ = Mstr;
elseif obj.appendM
    obj.M_ = [obj.M_ Mstr];
else
    % Replace the most recent entry with the same type / condition / area
    idx = (string([obj.M_.type]) == Mstr.type) & ...
          (string([obj.M_.condition]) == Mstr.condition) & ...
          (string([obj.M_.Area]) == Mstr.Area);
    if any(idx)
        obj.M_(idx) = Mstr;
    else
        obj.M_ = [obj.M_ Mstr];
    end
end
end
