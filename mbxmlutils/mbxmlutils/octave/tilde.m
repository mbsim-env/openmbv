function out=tilde(in)

  error('mfmf2');
  if (strcmp(typeinfo(in), 'swig_ref') && in.size1()==3 && in.size2()==1) || size(in)==[3,1]

    out=[     0, -in(3),  in(2) ;...
          in(3),      0, -in(1) ;...
         -in(2),  in(1),      0];

  elseif (strcmp(typeinfo(in), 'swig_ref') && in.size1()==3 && in.size2()==3) || size(in)==[3,3]

    if !strcmp(typeinfo(in), 'swig_ref')
      abserr=1e-7;
      if abs(in(1,1))>abserr || abs(in(2,2))>abserr || abs(in(3,3))>abserr || ...
         abs(in(1,2)+in(2,1))>abserr || abs(in(1,3)+in(3,1))>abserr || abs(in(2,3)+in(3,2))>abserr
        error('Must be s skew symmetric matrix.');
      end
    end
    out=[in(3,2); in(1,3); in(2,1)];

  else
    error('Must be called with with a 3x3 matrix or a column vector of length 3.');
  end
