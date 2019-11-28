function ret=horzcat(varargin)
  global MatrixSym;

  % helper functions
  function c=cols(x)
    if isscalar(x) && isfloat(x)
      c=1;
    elseif ismatrix(x) && isfloat(x)
      c=size(x,2);
    elseif strcmp(swig_type(x), 'IndependentVariable') || strcmp(swig_type(x), 'SymbolicExpression')
      c=1;
    elseif strcmp(swig_type(x), 'VectorIndep') || strcmp(swig_type(x), 'VectorSym')
      c=1;
    elseif strcmp(swig_type(x), 'MatrixSym')
      c=x.cols();
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

  % get/check number of rows
  nrrows=rows(varargin{1});
  for i=2:length(varargin)
    if nrrows~=rows(varargin{i})
      error('None matching number of rows in horzcat.');
    end
  end

  % get number of cols
  nrcols=zeros(length(varargin),1);
  for i=1:length(varargin)
    nrcols(i)=cols(varargin{i});
  end

  % create return value
  ret=MatrixSym(nrrows, sum(nrcols));
  for i=1:length(varargin)
    for r=1:nrrows
      for c=1:nrcols(i)
        ret(r,sum(nrcols(1:i-1))+c)=getElement(varargin{i},r,c);
      end
    end
  end
end
