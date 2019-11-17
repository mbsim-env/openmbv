function ret=horzcat(varargin)
  global swigGlobalModuleVar_fmatvec_symbolic_swig_octave;
  sym=swigGlobalModuleVar_fmatvec_symbolic_swig_octave;

  if isscalar(varargin{1}) && isfloat(varargin{1})
    rows=1;
  elseif ismatrix(varargin{1}) && isfloat(varargin{1}) && size(varargin{1},2)==1
    rows=size(varargin{1},1);
  elseif strcmp(swig_type(varargin{1}), 'IndependentVariable') || strcmp(swig_type(varargin{1}), 'SymbolicExpression')
    rows=1;
  elseif strcmp(swig_type(varargin{1}), 'VectorIndep') || strcmp(swig_type(varargin{1}), 'VectorSym')
    rows=varargin{1}.size();
  end
  cols=length(varargin);
  ret=sym.MatrixSym(rows, cols);
  for r=1:rows
    for c=1:cols
      x=varargin{c};
      if rows==1
        ret(r,c)=x;
      else
        ret(r,c)=x(r);
      end
    end
  end
end
