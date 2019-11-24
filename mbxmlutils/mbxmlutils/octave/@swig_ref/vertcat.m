function ret=vertcat(varargin)
  fmatvec_symbolic_swig_octave;

  % helper functions
  function [c,rettype]=cols(x)
    if isscalar(x) && isfloat(x)
      c=1;
      rettype='vector';
    elseif ismatrix(x) && isfloat(x)
      c=size(x,2);
      if c==1
        rettype='vector';
      else
        rettype='matrix';
      end
    elseif strcmp(swig_type(x), 'IndependentVariable') || strcmp(swig_type(x), 'SymbolicExpression')
      c=1;
      rettype='vector';
    elseif strcmp(swig_type(x), 'VectorIndep') || strcmp(swig_type(x), 'VectorSym')
      c=1;
      rettype='vector';
    elseif strcmp(swig_type(x), 'MatrixSym')
      c=x.cols();
      rettype='matrix';
    end
  end
  function r=rows(x)
    if isscalar(x) && isfloat(x)
      r=1;
    elseif ismatrix(x) && isfloat(x)
      r=size(x,1);
    elseif strcmp(swig_type(x), 'IndependentVariable') || strcmp(swig_type(x), 'SymbolicExpression')
      r=1;
    elseif strcmp(swig_type(x), 'VectorIndep') || strcmp(swig_type(x), 'VectorSym')
      r=x.size();
    elseif strcmp(swig_type(x), 'MatrixSym')
      r=x.rows();
    end
  end
  function r=getElement(x, r, c)
    if isscalar(x) && isfloat(x)
      r=x;
    elseif ismatrix(x) && isfloat(x)
      r=x(r,c);
    elseif strcmp(swig_type(x), 'IndependentVariable') || strcmp(swig_type(x), 'SymbolicExpression')
      r=x;
    elseif strcmp(swig_type(x), 'VectorIndep') || strcmp(swig_type(x), 'VectorSym')
      r=x(r);
    elseif strcmp(swig_type(x), 'MatrixSym')
      r=x(r,c);
    end
  end

  % get/check number of cols
  [c,rettype]=cols(varargin{1});
  nrcols=cols(varargin{1});
  for i=2:length(varargin)
    [c,rt]=cols(varargin{i});
    if nrcols~=c
      error('None matching number of cols in horzcat.');
    end
    if strcmp(rt, 'matrix')
      rettype='matrix';
    end
  end

  % get number of rows
  nrrows=zeros(length(varargin),1);
  for i=1:length(varargin)
    nrrows(i)=rows(varargin{i});
  end

  % create return value
  if strcmp(rt, 'matrix')
    ret=MatrixSym(sum(nrrows), nrcols);
  else
    ret=VectorSym(sum(nrrows));
  end
  for i=1:length(varargin)
    for r=1:nrrows(i)
      for c=1:nrcols
        if strcmp(rt, 'matrix')
          ret(sum(nrrows(1:i-1))+r,c)=getElement(varargin{i},r,c);
        else
          ret(sum(nrrows(1:i-1))+r)=getElement(varargin{i},r,c);
        end
      end
    end
  end
end
