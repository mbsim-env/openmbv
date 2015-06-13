function y=vertcat(varargin)
  global swigGlobalModuleVar_casadi_oct;
  I=size(varargin, 2);
  y=swigGlobalModuleVar_casadi_oct.SX.zeros(0,0);
  for i=1:I
    y.append(swigGlobalModuleVar_casadi_oct.SX(varargin{1, i}));
  end
end
