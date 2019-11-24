function A=euler(varargin)
  fmatvec_symbolic_swig_octave;

  nargs=length(varargin);
  if nargs==3
    PHI=varargin{1};
    theta=varargin{2};
    phi=varargin{3};
  elseif nargs==1
    PHI=varargin{1}(1);
    theta=varargin{1}(2);
    phi=varargin{1}(3);
  else
    error('Must be called with a three scalar arguments or one vector argument of length 3.');
  end

  A=[cos(phi)*cos(PHI)-sin(phi)*cos(theta)*sin(PHI),...
     -cos(phi)*cos(theta)*sin(PHI)-sin(phi)*cos(PHI),...
     sin(theta)*sin(PHI);
     cos(phi)*sin(PHI)+sin(phi)*cos(theta)*cos(PHI),...
     cos(phi)*cos(theta)*cos(PHI)-sin(phi)*sin(PHI),...
     -sin(theta)*cos(PHI);
     sin(phi)*sin(theta),...
     cos(phi)*sin(theta),...
     cos(theta)];
