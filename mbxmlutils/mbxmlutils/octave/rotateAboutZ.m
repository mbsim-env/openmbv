function A=rotateAboutZ(phi)
  if strcmp(typeinfo(phi), 'swig_ref')
    global swigGlobalModuleVar_fmatvec_symbolic_swig_octave;
    sym=swigGlobalModuleVar_fmatvec_symbolic_swig_octave;
    A=vertcat(horzcat(sym.cos(phi),-sym.sin(phi),0),
              horzcat(sym.sin(phi),sym.cos(phi),0),
              horzcat(0,0,1));
  else
    A=[cos(phi),-sin(phi),0;
       sin(phi),cos(phi),0;
       0,0,1];
  end
