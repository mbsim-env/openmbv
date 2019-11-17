function ret=vertcat(varargin)
  global swigGlobalModuleVar_fmatvec_symbolic_swig_octave;
  sym=swigGlobalModuleVar_fmatvec_symbolic_swig_octave;

  if isscalar(varargin{1}) && isfloat(varargin{1})
    cols=1;
  elseif ismatrix(varargin{1}) && isfloat(varargin{1}) && size(varargin{1},1)==1
    cols=size(varargin{1},2);
  elseif strcmp(swig_type(varargin{1}), 'IndependentVariable') || strcmp(swig_type(varargin{1}), 'SymbolicExpression')
    cols=1;
  elseif strcmp(swig_type(varargin{1}), 'MatrixSym') && varargin{1}.rows()==1
    cols=varargin{1}.cols();
  end

  rows=length(varargin);
  if cols==1
    ret=sym.VectorSym(rows);
    for i=1:rows
      ret(i)=varargin{i};
    end
  else
    ret=sym.MatrixSym(rows, cols);
    for r=1:rows
      x=varargin{r};
      for c=1:cols
        ret(r,c)=x(1,c);
      end
    end
  end
end
