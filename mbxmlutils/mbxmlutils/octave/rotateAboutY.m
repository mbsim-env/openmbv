function A=rotateAboutY(phi)
  fmatvec_symbolic_swig_octave;

  A=[cos(phi),0,sin(phi);
     [0,1,0];
     -sin(phi),0,cos(phi)];
