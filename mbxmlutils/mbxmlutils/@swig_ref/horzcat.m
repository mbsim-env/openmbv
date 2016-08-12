function y=horzcat(varargin)
  global swigGlobalModuleVar_casadi_oct;
  numArgs=size(varargin, 2);
  y=swigGlobalModuleVar_casadi_oct.SX.zeros(0,0);
  for i=1:numArgs
    y=swigGlobalModuleVar_casadi_oct.horzconcat_wrapper(y, swigGlobalModuleVar_casadi_oct.SX(varargin{1, i}));
  end
end
