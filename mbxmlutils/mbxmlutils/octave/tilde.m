function out=tilde(in)

  if size(in)==[3,1] || (strcmp(typeinfo(in), 'swig_ref') && (strcmp(swig_type(in), 'VectorIndep') || ...
                                                              strcmp(swig_type(in), 'VectorSym')))

    out=[0, -in(3),  in(2);...
         in(3),      0, -in(1);...
         -in(2),  in(1),      0];

  elseif size(in)==[3,3] || (strcmp(typeinfo(in), 'swig_ref') && strcmp(swig_type(in), 'MatrixSym'))

    out=[in(3,2); in(1,3); in(2,1)];

  else
    error('Must be called with with a 3x3 matrix or a column vector of length 3.');
  end
