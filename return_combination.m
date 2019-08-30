function varargout = return_combination(varargin)
if nargin == 1 && ~isstruct(varargin{1})
    error('Need more than just 1 input that is not a struct');
end

inp_has_param = any(cellfun(@ischar, varargin, 'uni', 1)); 
inp_is_struct = any(cellfun(@isstruct, varargin, 'uni', 1)); 

ip = inputParser; 
valid_vec = @(x) isvector(x) && isnumeric(x); 
default_num = -1; 
valid_num = @(x) isnumeric(x) && length(x) == 1; 
ip.addRequired('vec', valid_vec); 
ip.addParameter('num', default_num, valid_num);
ip.addParameter('label', {}, @iscell); 

if inp_has_param && inp_is_struct
    error('Cannot be having parameters and a first struct input together');
end

if inp_has_param && ~inp_is_struct 
    ip.parse(varargin{:});     
    vec = {ip.Results.vec}; 
    num = ip.Results.num;
    label = ip.Results.label;

    if num ~= default_num && ~isempty(label) && num ~= length(label)
        err_msg = ['Using one input struct vector and an additional parameter' ...
            ' needs to satisfy either of these following conditions'];
        error('%s\n%s\n%s\n%s\n', err_msg, ...
            '- Has just `num` to tell how many number of repetitions', ...
            '- Has just `label` to tell how the numer of repetitions, output as a struct', ...
            '- Redundantly has both, but their lengths need to be similar');
    end
    
    if num == default_num 
        num = length(label);
    end
end

if inp_is_struct && ~inp_has_param
    vec = cellfun(@(x) x.vec, varargin, 'uni', 0); 
    num = length(vec);
    label = cellfun(@(x) x.label, varargin, 'uni', 0);
end

if ~inp_is_struct && ~inp_has_param
    vec = varargin;
    num = length(vec); 
    label = {}; 
end

% TODO: assess if needed
% if ~all(cellfun(valid_vec, vec, 'uni', 1))
%     error('All vectors need to be numeric vectors');
% end
    
tmp_cell = cell(num,1);
[tmp_cell{:}] = ndgrid(vec{:}); 
tmp_cell = cellfun(@(x) x(:), tmp_cell, 'uni', 0);
combo_mat = horzcat(tmp_cell{:}); 

if nargout >= 1
    varargout{1} = combo_mat;
end 

if nargout >= 2
    varargout{2} = label; 
end

if nargout == 3
    varargout{3} = struct('combo', {combo_mat}, 'label', {label});
end

if nargout > 3 
    error('Too many output');
end
        
end