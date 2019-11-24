function A=rotateAboutZ(phi)
  fmatvec_symbolic_swig_octave;

  A=[cos(phi),-sin(phi),0;
     sin(phi),cos(phi),0;
     [0,0,1]];
