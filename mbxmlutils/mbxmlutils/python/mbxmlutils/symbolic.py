"""MBXMLUtils helper functions for symbolic handling.

MBXMLUtils's vector/matrix interface is based on Python's numpy ndmatrix but not on sympy Matrix.
Hence, a symbolic vector/matrix function using sympy must not use sympy Matrix but numpy ndarray.
This module provides several helper function to simplify this conversion.

Moreover, it provides helper functions to handle differentiation the same way as done in fmatvec:
- a scalar differentiated wrt a vector gives a row-vector
- a 3x3-rotation-matrix differentiated wrt to a scalar can automatically be converted to a 3-vector
- ..."""

import sympy
import numpy

def symbol(name, rows=None, cols=None):
  """Create a scalar (rows=None, cols=None), vector (cols=None) or matrix symbolic variable with basename name.
  For vector/matrix the element access is in fmatvec notation, e.g. name(idx) to access the 0-based idx-th element."""
  if rows is None and cols is None:
    return sympy.Symbol(name, real=True)
  if cols is None:
    return numpy.array(list(map(lambda r: sympy.Symbol(name+"("+str(r)+")", real=True), range(0,rows))))
  m=[]
  for r in range(0,rows):
    m.append(list(map(lambda c: sympy.Symbol(name+"("+str(r)+","+str(c)+")", real=True), range(0,cols))))
  return numpy.array(m)

def callPerComponent(x, func):
  """Calls func(comp) for each component in the scalar, vector or matrix variable x and return the result"""
  if not hasattr(x, "shape"):
    # scalar
    return func(x)
  elif len(x.shape)==1:
    # vector
    vec=[]
    for r in range(0, x.shape[0]):
      vec.append(func(x[r]))
    return numpy.array(vec)
  elif len(x.shape)==2:
    # matrix
    mat=[]
    for r in range(0, x.shape[0]):
      row=[]
      for c in range(0, x.shape[1]):
        row.append(func(x[r,c]))
      mat.append(row)
    return numpy.array(mat)
  else:
    raise RuntimeError("Unknonwn type: "+str(type(x)))

def diff(d, i, rotMat=False):
  """Calculate the derivative of each element in d with respect to the scalar i or vector i.

  The following combination of scalar/vector/matrix for d and i is allowed:
  d is        i is   -> result is
  -------------------------------
  scalar      scalar -> scalar
  vector      scalar -> vector
  rowvector   scalar -> rowvector
  matrix      scalar -> matrix
  scalar      vector -> rowvector
  vector      vector -> matrix

  If rotMat==True and d is a 3x3-matrix then also the following combinations are allowed:
  d is        i is   -> result is
  --------------------------------
  matrix      scalar -> 3-vector
  matrix      vector -> 3xN-vector   (with N = len(i))
  See https://www.mbsim-env.de/base/fileDownloadFromDB/service/Manual/rotmatder/manualFile/ for details
  """
  if not hasattr(i, "shape"):
    # i is scalar
    der=callPerComponent(d, lambda e: sympy.diff(e, i))
    if not rotMat:
      return der
    else:
      import mbxmlutils
      if d.shape!=(3,3):
        raise RuntimeError("diff with rotMat=True can only be called with a 3x3 matrix")
      return mbxmlutils.tilde(der @ d.T)
  elif len(i.shape)==1:
    if len(d.shape)==2 and not rotMat:
      raise RuntimeError("diff with with a vector i can only be called with a scalar or vector for d")
    # i is vector
    der=[]
    for ii in i:
      der.append(diff(d, ii))
    if not rotMat:
      return numpy.stack(der, axis=1)
    else:
      import mbxmlutils
      ret=[]
      for x in der:
        ret.append(mbxmlutils.tilde(x @ d.T))
      return numpy.stack(ret, axis=1)
  else:
    raise RuntimeError("Unknonwn type: "+str(type(i)))

def diffDir(d, i, di, rotMat=False):
  """Calculate the directional derivative of d with respect to i in the direction of di, see also diff.
  """
  if not hasattr(i, "shape"):
    # i is scalar
    i = [i]
    di = [di]
  elif len(i.shape)==1:
    # i is vector
    pass
  else:
    raise RuntimeError("Unknonwn type: "+str(type(i)))
  firstCall = True
  for idx in range(0, len(i)):
    if firstCall:
      firstCall = False
      ret = diff(d, i[idx], rotMat=rotMat) * di[idx]
    else:
      ret += diff(d, i[idx], rotMat=rotMat) * di[idx]
  return ret

def subs(x, *args):
  """Substitude all components of x"""
  def _subs(e, *args):
    if hasattr(e, "subs"):
      return e.subs(*args)
    else:
      return e
  return callPerComponent(x, lambda e: _subs(e,*args))

def float_(x):
  """evaluate each element to float"""
  return callPerComponent(x, lambda e: float(e))

def C(x, asVectorIfPossible=True):
  """Convert from numpy to sympy scalar/vector/matrix and the other way around.
  Note that a sympy matrix of size Nx1 is converted to a numpy vector of shape (N,) if asVectorIfPossible=True
  (there is no sympy vector). Set asVectorIfPossible==False to avoid this and convert to a Nx1 matrix in numpy."""
  if type(x)==numpy.ndarray:
    # convert numpy array to sympy matrix
    return sympy.Matrix(x)
  if x.__class__.__name__=="MutableDenseMatrix" or x.__class__.__name__=="ImmutableDenseMatrix" or x.__class__.__name__=="ImmutableDenseNDimArray":
    # convert sympy matrix to numpy array (as 1D or 2D)
    if asVectorIfPossible and x.shape[1]==1:
      return numpy.reshape(x, (x.shape[0],))
    return numpy.array(x)
  # no conversion needed for scalars
  return x

def ccode(*xx, retName="ret", subsName="x", oneOutputPerLine=False, cse=True):
  """Generate c-code for all expressions *xx.
  If only one expression is provided use retName as output variable name.
  If more then one expression is provided and retName is a string use "{retName}[{idx}]" as output variable name.
  If more then one expression is provided and retName is a list of string use "{retName[idx]] as output variable name.
  If common subexpressions are extracted (cse=True) then subsName is used as prefix for cse variables.
  If oneOutputPerLine=True matrix expressions a printed one line per element, else matrices are printed as a matrix."""
  if cse:
    # convert matrix/vector to sympy
    sympy_xx = []
    for x in xx:
      if type(x)==numpy.ndarray:
        sympy_xx.append(sympy.Matrix(x))
      else:
        sympy_xx.append(x)
    # extract common subexpressions
    (subs,sympy_xx_cse)=sympy.cse(sympy_xx, symbols=sympy.numbered_symbols(subsName))
    # convert sympy_xx_cse back to numpy (while keeping the original scalar/vector/matrix form)
    numpy_xx_cse = []
    for (x,xorg) in zip(sympy_xx_cse, xx):
      if x.__class__.__name__=="MutableDenseMatrix" or x.__class__.__name__=="ImmutableDenseMatrix":
        numpy_xx_cse.append(numpy.reshape(x, xorg.shape))
      else:
        numpy_xx_cse.append(x)
  else:
    # do not extract cse (no subs and numpy_xx_cse=xx)
    subs=[]
    numpy_xx_cse=xx

  def dumpScalar(x):
    x = x.evalf()
    return sympy.ccode(x)

  fullRet=""
  
  # dump common subexpressions
  for x in subs:
    fullRet += f"double {x[0]} = {dumpScalar(x[1])};\n"

  # dump the expressions
  exprIdx=-1
  for x in numpy_xx_cse:
    exprIdx+=1
    # expression output variable name
    if len(xx)>1:
      if type(retName) == list:
        retNameFull=f"{retName[exprIdx]}"
      else:
        retNameFull=f"{retName}[{exprIdx}]"
    else:
      retNameFull=f"{retName}"

    if not hasattr(x, "shape"):
      # dump scalar
      fullRet += f"{retNameFull} = {dumpScalar(x)};\n"
    elif len(x.shape)==1:
      # dump vector
      lr=len(str(x.shape[0]-1))
      ret=""
      for r in range(0, x.shape[0]):
        ret+=f"{retNameFull}({r:{lr}}) = {dumpScalar(x[r])};\n"
      fullRet += ret
    elif len(x.shape)==2:
      # dump matrix
      lr=len(str(x.shape[0]-1))
      lc=len(str(x.shape[1]-1))
      l=numpy.zeros((x.shape[1],))
      mat=[]
      # calculate column size
      for r in range(0, x.shape[0]):
        row=[]
        for c in range(0, x.shape[1]):
          code=dumpScalar(x[r,c])
          row.append(code)
          l[c]=max(l[c], len(code))
        mat.append(row)
      ret=""
      # dump
      for r in range(0, x.shape[0]):
        for c in range(0, x.shape[1]):
          if oneOutputPerLine:
            ret+=f"{retNameFull}({r:{lr}},{c:{lc}}) = {mat[r][c]};\n"
          else:
            ret+=f"{retNameFull}({r:{lr}},{c:{lc}}) = {mat[r][c].ljust(int(l[c]))};"+(" " if c<x.shape[1]-1 else "")
        if not oneOutputPerLine:
          ret+="\n"
      fullRet += ret
  return fullRet
