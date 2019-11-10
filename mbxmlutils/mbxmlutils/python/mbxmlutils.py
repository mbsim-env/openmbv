import numpy
import sympy
import uuid



def _convertScalar(x):
  if hasattr(x, "free_symbols") and len(x.free_symbols)==0:
    return float(x)
  else:
    return x

def _convert(x):
  if hasattr(x, '__len__'):
    return numpy.array([_convert(xi) for xi in x])
  else:
    return _convertScalar(x)

def _serializeFunction(x):
  if x.__class__.__name__=="MutableDenseMatrix" or x.__class__.__name__=="ImmutableDenseMatrix":
    s="["
    (rows, cols)=x.shape;
    for r in range(0, rows):
      for c in range(0, cols):
        s+=_serializeFunction(x[r,c])
        if c<cols-1:
          s+=", "
      if r<rows-1:
        s+="; "
    s+="]"
    return s
  else:
    def serializeVertex(x):
      if x.func.__name__=="Symbol":
        uid=_serializeFunction.indepMap.setdefault(hash(x), uuid.uuid4())
        return " s "+str(uid)
      if isinstance(x, sympy.Integer):
        return " i "+str(x)
      if isinstance(x, sympy.Float) or isinstance(x, sympy.Rational):
        return " d "+str(x)
      opStr=_serializeFunction.opMap.get(x.func.__name__)
      if opStr==None:
        raise RuntimeError("Unknown operator "+x.func.__name__+": "+str(x))
      nrArgs=len(x.args)
      if (x.func.__name__=="Add" or x.func.__name__=="Mul") and nrArgs>2:
        return serializeVertex(x.func(x.args[0], x.func(*x.args[1:]), evaluate=False));
      if opStr[1]!=nrArgs:
        raise RuntimeError("Number of arguments of operator "+x.func.__name__+" does not match: "+str(x))
      s=" o "+opStr[0]+" "+str(nrArgs)
      for op in x.args:
        s+=serializeVertex(op)
      return s
    s="{"
    s+=serializeVertex(x)
    s+=" }"
    return s
_serializeFunction.indepMap={}
_serializeFunction.opMap={
  'Add':   ("+", 2),
# '':      ("-", 2),
  'Mul':   ("*", 2),
# '':      ("/", 2),
  'Pow':   ("pow", 2),
  'log':   ("log", 1),
  'sqrt':  ("sqrt", 1),
# '':      ("neg", 1),
  'sin':   ("sin", 1),
  'cos':   ("cos", 1),
  'tan':   ("tan", 1),
  'sinh':  ("sinh", 1),
  'cosh':  ("cosh", 1),
  'tanh':  ("tanh", 1),
  'asin':  ("asin", 1),
  'acos':  ("acos", 1),
  'atan':  ("atan", 1),
  'asinh': ("asinh", 1),
  'acosh': ("acosh", 1),
  'atanh': ("atanh", 1),
}



def registerPath(path):
  # load the libmbxmlutils-eval-global-python.so ones
  if registerPath.dll==None:
    import ctypes
    import sys
    if sys.platform=='linux2':
      registerPath.dll=ctypes.cdll.LoadLibrary("libmbxmlutils-eval-global-python.so")
    else:
      registerPath.dll=ctypes.cdll.LoadLibrary("libmbxmlutils-eval-global-python")
  # call the mbxmlutilsPyEvalRegisterPath function in this lib
  ret=registerPath.dll.mbxmlutilsPyEvalRegisterPath(path)
registerPath.dll=None



def load(filename):
  import csv
  with open(filename, 'r') as fileObj:
    reader=csv.reader(fileObj, delimiter=' ')
    ret=None
    for row in reader:
      if ret is None:
        ret=numpy.zeros((0, len(row)))
      ret=numpy.append(ret, [list(map(float, row))], 0)
  return ret



def cardan(*argv):
  if len(argv)==3:
    alpha=argv[0]
    beta=argv[1]
    gamma=argv[2]
  elif len(argv)==1 and len(argv[0])==3:
    alpha=argv[0][0];
    beta=argv[0][1];
    gamma=argv[0][2];
  else:
    raise RuntimError('Must be called with a three scalar arguments or one vector argument of length 3.')

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
