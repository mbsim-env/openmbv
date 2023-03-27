import numpy
import sympy
import uuid
import os
if list(map(lambda x: int(x), sympy.__version__.split(".")[0:2])) >= [1,6]:
  sympyRelational=sympy.core.relational
else:
  sympyRelational=sympy.relational



# INTERNAL FUNCTION
# convert the scalar x to float values if possible
def _convertScalar(x):
  if hasattr(x, "free_symbols") and len(x.free_symbols)==0:
    return float(x)
  else:
    return x

# INTERNAL FUNCTION
# convert all elements of the matrix, vector or scalar x to float values if possible
def _convert(x):
  if hasattr(x, '__len__'):
    return numpy.array([_convert(xi) for xi in x])
  else:
    return _convertScalar(x)

# INTERNAL FUNCTION
# serialize a symbolic expression to the fmatvec format
def _serializeFunction(x):
  # serialize a matrix or vector
  if x.__class__.__name__=="MutableDenseMatrix" or x.__class__.__name__=="ImmutableDenseMatrix":
    s="["
    (rows, cols)=x.shape
    for r in range(0, rows):
      for c in range(0, cols):
        s+=_serializeFunction(x[r,c])
        if c<cols-1:
          s+=", "
      if r<rows-1:
        s+="; "
    s+="]"
    return s
  # serialize a scalar
  else:
    # serialize a scalar (may be called recursively)
    def serializeVertex(x):
      # serialize a integer
      if isinstance(x, sympy.Integer) or numpy.issubdtype(type(x), int) or isinstance(x, int):
        return str(x)
      # serialize a float
      if isinstance(x, sympy.Float) or numpy.issubdtype(type(x), float) or isinstance(x, sympy.Rational) or isinstance(x, float):
        return str(x)
      # serialize a independent variable (all independe variables must be sympy.Dummy classes
      if x.func.__name__=="Dummy":
        # create a unique UUID for the variable (map the hash to a UUID)
        uid=_serializeFunction.indepMap.setdefault(hash(x), uuid.uuid4())
        # for debug runs (FMATVEC_DEBUG_SYMBOLICEXPRESSION_UUID=1) map the UUID to a counting int and use it
        # (to generate equal results on each run)
        if _serializeFunction.FMATVEC_DEBUG_SYMBOLICEXPRESSION_UUID is not None:
          nr=_serializeFunction.mapUUIDInt.setdefault(uuid, len(_serializeFunction.mapUUIDInt)+1)
          return "s"+str(nr)
        # for release runs use the UUID
        else:
          return str(uid)
      # sympy.UnevaluatedExpr are just noops
      if x.func.__name__=="UnevaluatedExpr":
        return serializeVertex(x.args[0])
      # sympy.Piecewise is handled specially
      if x.func.__name__=="Piecewise":
        nrArgs=len(x.args)
        # convert a single piecewise function with multiple arguments to multiple piecwise functions with only one argument (condition)
        if nrArgs>1 and x.args[1][1]!=True:
          return serializeVertex(sympy.Piecewise(x.args[0], (sympy.UnevaluatedExpr(sympy.Piecewise(*x.args[1:])), True)))
        # serialize the piecewise function to the correspoding fmatvec representation
        else:
          if nrArgs>2:
            raise RuntimeError("Internal error in: "+str(x))
          c=x.args[0][1]
          gt=x.args[0][0]
          le=x.args[1][0] if nrArgs==2 else 0
          if nrArgs==2 and x.args[1][1]!=True:
            raise RuntimeError("Internal error in: "+str(x))
          if isinstance(c, sympyRelational.GreaterThan) or isinstance(c, sympyRelational.StrictGreaterThan):
            return "condition("+serializeVertex(c.lhs-c.rhs)+","+serializeVertex(gt)+","+serializeVertex(le)+")"
          elif isinstance(c, sympyRelational.LessThan) or isinstance(c, sympyRelational.StrictLessThan):
            return "condition("+serializeVertex(c.rhs-c.lhs)+","+serializeVertex(gt)+","+serializeVertex(le)+")"
          elif c==True:
            return serializeVertex(gt)
          elif c==False:
            return serializeVertex(le)
          else:
            raise RuntimeError("Unknown relation in Piecewise: "+str(x))
      # handle known function from opMap
      opStr=_serializeFunction.opMap.get(x.func.__name__)
      if opStr is None:
        raise RuntimeError("Unknown operator "+x.func.__name__+": "+str(x))
      nrArgs=len(x.args)
      # special handling of Add/Mul/Min/Max: convert more than 2 args into multiple function with just 2 args
      if (x.func.__name__=="Add" or x.func.__name__=="Mul" or x.func.__name__=="Min" or x.func.__name__=="Max") and nrArgs>2:
        return serializeVertex(x.func(x.args[0], sympy.UnevaluatedExpr(x.func(*x.args[1:]))))
      # check if the number of arguments match
      if not (
        opStr[1]==nrArgs or # number of arguments must match OR
        (x.func.__name__=="Heaviside" and nrArgs==2) # the Heaviside can also have a second default argument
        ):
        raise RuntimeError("Number of arguments of operator "+x.func.__name__+" does not match: "+str(x))
      # serailize
      s=opStr[0]+"("
      first=True
      for op in x.args[0:opStr[1]]:
        if not first: s+=","
        s+=serializeVertex(op)
        first=False
      s+=")"
      return s
    # serialize x to string s (may call serializeVertex recursively)
    s=serializeVertex(x)
    return s
_serializeFunction.indepMap={}
_serializeFunction.mapUUIDInt={}
_serializeFunction.FMATVEC_DEBUG_SYMBOLICEXPRESSION_UUID=os.getenv("FMATVEC_DEBUG_SYMBOLICEXPRESSION_UUID")
_serializeFunction.opMap={
  'Add':       ("plus", 2),
# '':          ("minus", 2),
  'Mul':       ("mult", 2),
# '':          ("div", 2),
  'Pow':       ("pow", 2),
  'log':       ("log", 1),
  'sqrt':      ("sqrt", 1),
# '':          ("neg", 1),
  'sin':       ("sin", 1),
  'cos':       ("cos", 1),
  'tan':       ("tan", 1),
  'sinh':      ("sinh", 1),
  'cosh':      ("cosh", 1),
  'tanh':      ("tanh", 1),
  'asin':      ("asin", 1),
  'acos':      ("acos", 1),
  'atan':      ("atan", 1),
  'atan2':     ("atan2", 2),
  'asinh':     ("asinh", 1),
  'acosh':     ("acosh", 1),
  'atanh':     ("atanh", 1),
  'exp':       ("exp", 1),
  'sign':      ("sign", 1),
  'Heaviside': ("heaviside", 1),
  'Abs':       ("abs", 1),
  'Min':       ("min", 2),
  'Max':       ("max", 2),
}



def registerPath(path):
  # load the libmbxmlutils-eval-global-python.so ones
  if registerPath.dll is None:
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
      registerPath.dll=ctypes.cdll.LoadLibrary("libmbxmlutils-eval-global-python.so")
    else:
      registerPath.dll=ctypes.cdll.LoadLibrary("libmbxmlutils-eval-global-python")
  # call the mbxmlutilsPyEvalRegisterPath function in this lib
  registerPath.dll.mbxmlutilsPyEvalRegisterPath(path.encode("utf-8"))
registerPath.dll=None



def load(filename):
  return numpy.genfromtxt(filename)



def cardan(*argv):
  if len(argv)==3:
    alpha=argv[0]
    beta=argv[1]
    gamma=argv[2]
  elif len(argv)==1 and len(argv[0])==3:
    alpha=argv[0][0]
    beta=argv[0][1]
    gamma=argv[0][2]
  else:
    raise RuntimeError('Must be called with a three scalar arguments or one vector argument of length 3.')

  return _convert(numpy.array([[sympy.cos(beta)*sympy.cos(gamma),
                              -sympy.cos(beta)*sympy.sin(gamma),
                              sympy.sin(beta)],
                              [sympy.cos(alpha)*sympy.sin(gamma)+sympy.sin(alpha)*sympy.sin(beta)*sympy.cos(gamma),
                              sympy.cos(alpha)*sympy.cos(gamma)-sympy.sin(alpha)*sympy.sin(beta)*sympy.sin(gamma),
                              -sympy.sin(alpha)*sympy.cos(beta)],
                              [sympy.sin(alpha)*sympy.sin(gamma)-sympy.cos(alpha)*sympy.sin(beta)*sympy.cos(gamma),
                              sympy.cos(alpha)*sympy.sin(beta)*sympy.sin(gamma)+sympy.sin(alpha)*sympy.cos(gamma),
                              sympy.cos(alpha)*sympy.cos(beta)]]))



def euler(*argv):
  if len(argv)==3:
    PHI=argv[0]
    theta=argv[1]
    phi=argv[2]
  elif len(argv)==1 and len(argv[0])==3:
    PHI=argv[0][0]
    theta=argv[0][1]
    phi=argv[0][2]
  else:
    raise RuntimeError('Must be called with a three scalar arguments or one vector argument of length 3.')

  return _convert(numpy.array([[sympy.cos(phi)*sympy.cos(PHI)-sympy.sin(phi)*sympy.cos(theta)*sympy.sin(PHI),
                              -sympy.cos(phi)*sympy.cos(theta)*sympy.sin(PHI)-sympy.sin(phi)*sympy.cos(PHI),
                              sympy.sin(theta)*sympy.sin(PHI)],
                              [sympy.cos(phi)*sympy.sin(PHI)+sympy.sin(phi)*sympy.cos(theta)*sympy.cos(PHI),
                              sympy.cos(phi)*sympy.cos(theta)*sympy.cos(PHI)-sympy.sin(phi)*sympy.sin(PHI),
                              -sympy.sin(theta)*sympy.cos(PHI)],
                              [sympy.sin(phi)*sympy.sin(theta),
                              sympy.cos(phi)*sympy.sin(theta),
                              sympy.cos(theta)]]))



def rotateAboutX(phi):
  return _convert(numpy.array([[1,0,0],
                              [0,sympy.cos(phi),-sympy.sin(phi)],
                              [0,sympy.sin(phi),sympy.cos(phi)]]))



def rotateAboutY(phi):
  return _convert(numpy.array([[sympy.cos(phi),0,sympy.sin(phi)],
                              [0,1,0],
                              [-sympy.sin(phi),0,sympy.cos(phi)]]))



def rotateAboutZ(phi):
  return _convert(numpy.array([[sympy.cos(phi),-sympy.sin(phi),0],
                              [sympy.sin(phi),sympy.cos(phi),0],
                              [0,0,1]]))



def tilde(x):

  if len(x)==3 and not hasattr(x[0], '__len__'):

    return numpy.array([[    0, -x[2],  x[1]], \
                        [ x[2],     0, -x[0]], \
                        [-x[1],  x[0],     0]])

  elif len(x)==3 and len(x[0])==3 and not hasattr(x, 'shape'):

    return numpy.array([x[2][1], x[0][2], x[1][0]])

  elif hasattr(x, 'shape') and x.shape==(3,3):

    return numpy.array([x[2,1], x[0,2], x[1,0]])

  else:
    raise RuntimeError('Must be called with with a 3x3 matrix or a column vector of length 3.')
