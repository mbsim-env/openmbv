function A=rotateAboutX(phi)
  if strcmp(typeinfo(phi), 'swig_ref')
    global swigGlobalModuleVar_fmatvec_symbolic_swig_octave;
    sym=swigGlobalModuleVar_fmatvec_symbolic_swig_octave;
    A=vertcat(horzcat(1,0,0),
              horzcat(0,sym.cos(phi),-sym.sin(phi)),
              horzcat(0,sym.sin(phi),sym.cos(phi)));
  else
    A=[1,0,0;
       0,cos(phi),-sin(phi);
       0,sin(phi),cos(phi)];
  end
