function y=vertcat(varargin)
  global casadi;
  I=size(varargin, 2);
  y=casadi.SX.zeros(0,0);
  for i=1:I
    y.append(casadi.SX(varargin{1, i}));
  end
end
