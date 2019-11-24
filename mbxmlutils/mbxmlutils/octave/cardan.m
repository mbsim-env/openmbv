function A=cardan(varargin)
  fmatvec_symbolic_swig_octave;

  nargs=length(varargin);
  if nargs==3
    alpha=varargin{1};
    beta=varargin{2};
    gamma=varargin{3};
  elseif nargs==1
    alpha=varargin{1}(1);
    beta=varargin{1}(2);
    gamma=varargin{1}(3);
  else
    error('Must be called with a three scalar arguments or one vector argument of length 3.');
  end

  A=[cos(beta)*cos(gamma),...
     -cos(beta)*sin(gamma),...
     sin(beta);
     cos(alpha)*sin(gamma)+sin(alpha)*sin(beta)*cos(gamma),...
     cos(alpha)*cos(gamma)-sin(alpha)*sin(beta)*sin(gamma),...
     -sin(alpha)*cos(beta);
     sin(alpha)*sin(gamma)-cos(alpha)*sin(beta)*cos(gamma),...
     cos(alpha)*sin(beta)*sin(gamma)+sin(alpha)*cos(gamma),...
     cos(alpha)*cos(beta)];
