function A=rotateAboutX(phi)
  if strcmp(typeinfo(phi), 'swig_ref')
    casadi=swigLocalLoad('casadi_oct'); % a workaround since vertcat does now work with a pure none casadi row
    one=casadi.SX(1);
  else
    one=1;
  end
  A=[one,0,0;
     0,cos(phi),-sin(phi);
     0,sin(phi),cos(phi)];
