function A=rotateAboutY(phi)
  if strcmp(typeinfo(phi), 'swig_ref')
    global swigGlobalModuleVar_fmatvec_symbolic_swig_octave;
    sym=swigGlobalModuleVar_fmatvec_symbolic_swig_octave;
    A=vertcat(horzcat(sym.cos(phi),0,sym.sin(phi)),
              horzcat(0,1,0),
              horzcat(-sym.sin(phi),0,sym.cos(phi)));
  else
    A=[cos(phi),0,sin(phi);
       0,1,0;
       -sin(phi),0,cos(phi)];
  end
