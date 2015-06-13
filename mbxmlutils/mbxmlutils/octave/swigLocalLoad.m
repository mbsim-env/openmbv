% Load the SWIG module named moduleName locally and return the SWIG module variable
function swigModuleVar=swigLocalLoad(moduleName)
  global SWIG_autoload_handling;
  % enable SWIG autoload handling
  SWIG_autoload_handling=1;
  % load swig module
  try
    eval([moduleName ';']);
  catch
    % reset to normal autoload handling
    SWIG_autoload_handling=0;
    error(lasterr);
  end
  % reset to normal autoload handling
  SWIG_autoload_handling=0;
  % return swig module variable (as local namespace)
  swigModuleVar=eval(moduleName);
  % save module as global variable
  eval(['global swigGlobalModuleVar_' moduleName ';']);
  eval(['swigGlobalModuleVar_' moduleName '=swigModuleVar;']);
end

% overloaded 'autoload' function which behavies like the builtin function except
% when called from swigLocalLoad.
function ret=autoload(func, file)
  global SWIG_autoload_handling;
  % add these functions always (independent of the SWIG_autoload_handling flag)
  alwaysAdd={'^swig_type$', '^swig_typequery$', '^swig_this$', '^subclass$', '^op_.*$'};
  % call the builtin function (original function) only if ...
  if ~SWIG_autoload_handling || ~exist('func') || size(cell2mat(regexp(func, alwaysAdd)), 2)>0
    if nargout==1
      ret=builtin('autoload');
    else
      builtin('autoload', func, file);
    end
  end
  % ... else do nothing
end
