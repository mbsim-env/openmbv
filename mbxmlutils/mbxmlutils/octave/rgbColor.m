function hsv=rgbColor(varargin)

  nargs=length(varargin);
  if nargs==3
    red=varargin{1};
    green=varargin{2};
    blue=varargin{3};
  elseif nargs==1
    red=varargin{1}(1);
    green=varargin{1}(2);
    blue=varargin{1}(3);
  else
    error('Must be called with a three scalar arguments or one vector argument of length 3.');
  end

  hsv=rgb2hsv([red,green,blue])'
