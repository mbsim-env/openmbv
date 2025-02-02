import os
import numpy



# INTERNAL FUNCTION
# wrap math or sympy functions: if all args are int or float use math else use sympy
# (we try to avoid importing sympy whenever possible since sympy has a quite significant load time)
class _MathOrSympy:
  def __init__(self, *t):
    self.useMath=all(map(lambda x: not hasattr(x, "free_symbols") or (hasattr(x, "free_symbols") and len(x.free_symbols)==0), t))
  def Min(self, *x):
    if self.useMath:
      return min(*x)
    else:
      import sympy
      return sympy.Min
  def Max(self, *x):
    if self.useMath:
      return max(*x)
    else:
      import sympy
      return sympy.Max
  def Abs(self, *x):
    if self.useMath:
      return abs(*x)
    else:
      import sympy
      return sympy.Abs
  def __getattr__(self, attr):
    if self.useMath:
      import math
      return getattr(math, attr)
    else:
      import sympy
      return getattr(sympy, attr)

# INTERNAL FUNCTION
# convert the scalar x to float values if possible
def _convertScalar(x):
  if not hasattr(x, "free_symbols") or (hasattr(x, "free_symbols") and len(x.free_symbols)==0):
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
  # serialize a matrix or vector (we allow sympy matrices, numpy array and python lists)
  if x.__class__.__name__=="MutableDenseMatrix" or x.__class__.__name__=="ImmutableDenseMatrix" or \
     type(x)==numpy.ndarray:
    s="["
    if len(x.shape)==2:
      (rows, cols)=x.shape
      for r in range(0, rows):
        for c in range(0, cols):
          s+=_serializeFunction(x[r,c])
          if c<cols-1:
            s+=", "
        if r<rows-1:
          s+="; "
    elif len(x.shape)==1:
      rows=x.shape[0]
      for r in range(0, rows):
        s+=_serializeFunction(x[r])
        if r<rows-1:
          s+="; "
    else:
      raise RuntimeError("Internal error: Unknonw shape")
    s+="]"
    return s
  elif type(x)==list:
    s="["
    if len(x)>0 and type(x[0])==list:
      rows=len(x)
      for r in range(0, rows):
        cols=len(x[r])
        for c in range(0, cols):
          s+=_serializeFunction(x[r][c])
          if c<cols-1:
            s+=", "
        if r<rows-1:
          s+="; "
    elif (len(x)>0 and type(x[0])!=list) or len(x)==0:
      rows=len(x)
      for r in range(0, rows):
        s+=_serializeFunction(x[r])
        if r<rows-1:
          s+="; "
    else:
      raise RuntimeError("Internal error: Unknonw shape")
    s+="]"
    return s
  # serialize a scalar
  else:
    # serialize a scalar (may be called recursively)
    def serializeVertex(x):
      import sympy
      # serialize a integer
      if isinstance(x, sympy.Integer) or numpy.issubdtype(type(x), int) or isinstance(x, int):
        return str(int(x))
      # serialize a float
      if isinstance(x, sympy.Float) or numpy.issubdtype(type(x), float) or isinstance(x, sympy.Rational) or isinstance(x, float):
        return str(float(x))
      # serialize a independent variable (all independe variables must be sympy.Dummy classes
      if x.func.__name__=="Dummy":
        import uuid
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

          if list(map(lambda x: int(x), sympy.__version__.split(".")[0:2])) >= [1,6]:
            sympyRelational=sympy.core.relational
          else:
            sympyRelational=sympy.relational

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



def _getDLL():
  # load the libmbxmlutils-eval-global-python.so ones
  if _getDLL.dll is None:
    import ctypes
    import sys
    if sys.platform.startswith('linux'):
      _getDLL.dll=ctypes.cdll.LoadLibrary("libmbxmlutils-eval-global-python.so")
    else:
      _getDLL.dll=ctypes.cdll.LoadLibrary("libmbxmlutils-eval-global-python")
  return _getDLL.dll
_getDLL.dll=None



def registerPath(path):
  # call the mbxmlutils_Eval_registerPath function from the lib of the _getDLL call
  import ctypes
  _getDLL().mbxmlutils_Eval_registerPath.argtypes=[ctypes.c_char_p]
  _getDLL().mbxmlutils_Eval_registerPath(path.encode("utf-8"))



# return the (original) filename which contains the currently evaluated element
def getOriginalFilename():
  # call the mbxmlutils_DOMElement_getOriginalFilename function from the lib of the _getDLL call
  import ctypes
  _getDLL().mbxmlutils_DOMElement_getOriginalFilename.restype=ctypes.c_char_p
  return _getDLL().mbxmlutils_DOMElement_getOriginalFilename().decode("utf-8")



def installPrefix():
  return os.path.realpath(os.path.dirname(__file__)+"/../../../..")



def load(filename):
  return numpy.genfromtxt(filename)
def mbxmlutils_load(filename):
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
  ms=_MathOrSympy(alpha,beta,gamma)

  return _convert(numpy.array([[ms.cos(beta)*ms.cos(gamma),
                              -ms.cos(beta)*ms.sin(gamma),
                              ms.sin(beta)],
                              [ms.cos(alpha)*ms.sin(gamma)+ms.sin(alpha)*ms.sin(beta)*ms.cos(gamma),
                              ms.cos(alpha)*ms.cos(gamma)-ms.sin(alpha)*ms.sin(beta)*ms.sin(gamma),
                              -ms.sin(alpha)*ms.cos(beta)],
                              [ms.sin(alpha)*ms.sin(gamma)-ms.cos(alpha)*ms.sin(beta)*ms.cos(gamma),
                              ms.cos(alpha)*ms.sin(beta)*ms.sin(gamma)+ms.sin(alpha)*ms.cos(gamma),
                              ms.cos(alpha)*ms.cos(beta)]]))



def invCardan(T):
  ms=_MathOrSympy(T[0][2], T[1][2], T[2][2], T[0][1], T[0][0], T[1][0], T[1][1])
  if T[0][2]<-1-1e-12 or T[0][2]>1+1e-12:
    raise RuntimeError("Argument of invCardan is not a rotation matrix (due to numerical errors)")
  beta=ms.asin(ms.Max(ms.Min(T[0][2], 1.0), -1.0))
  nenner=ms.cos(beta)
  if ms.Abs(nenner)>1e-10:
    alpha=ms.atan2(-T[1][2],T[2][2])
    gamma=ms.atan2(-T[0][1],T[0][0])
  else:
    alpha=0
    gamma=ms.atan2(T[1][0],T[1][1])
  return _convert(numpy.array([alpha, beta, gamma]))



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
  ms=_MathOrSympy(PHI,theta,phi)

  return _convert(numpy.array([[ms.cos(phi)*ms.cos(PHI)-ms.sin(phi)*ms.cos(theta)*ms.sin(PHI),
                              -ms.cos(phi)*ms.cos(theta)*ms.sin(PHI)-ms.sin(phi)*ms.cos(PHI),
                              ms.sin(theta)*ms.sin(PHI)],
                              [ms.cos(phi)*ms.sin(PHI)+ms.sin(phi)*ms.cos(theta)*ms.cos(PHI),
                              ms.cos(phi)*ms.cos(theta)*ms.cos(PHI)-ms.sin(phi)*ms.sin(PHI),
                              -ms.sin(theta)*ms.cos(PHI)],
                              [ms.sin(phi)*ms.sin(theta),
                              ms.cos(phi)*ms.sin(theta),
                              ms.cos(theta)]]))



def rotateAboutX(phi):
  ms=_MathOrSympy(phi)
  return _convert(numpy.array([[1,0,0],
                              [0,ms.cos(phi),-ms.sin(phi)],
                              [0,ms.sin(phi),ms.cos(phi)]]))



def rotateAboutY(phi):
  ms=_MathOrSympy(phi)
  return _convert(numpy.array([[ms.cos(phi),0,ms.sin(phi)],
                              [0,1,0],
                              [-ms.sin(phi),0,ms.cos(phi)]]))



def rotateAboutZ(phi):
  ms=_MathOrSympy(phi)
  return _convert(numpy.array([[ms.cos(phi),-ms.sin(phi),0],
                              [ms.sin(phi),ms.cos(phi),0],
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



# Apply the steiner rule on a inertia.
#
# If the input "inertia" is a inertia tensor around its center of mass then the
# output is the corresponding inertia tensor around a point "com" away from the center of mass (note that the sign of "com" does not care: com and -com results in the same).
# "mass", which must be positive in this case, is the mass of the object.
#
# If the input "inertia" is a inertia tensor around a point "com" away from the center of mass then the
# output is the corresponding inertia tensor around the center of mass (note that the sign of "com" does not care: com and -com results in the same).
# "mass", which must be negative, in this case, is the negative mass of the object.
#
# "inertiaOrientation" is optional, if given, "inertia" is not with respect to the local system L (in which com is given) but with respect to a system K
# and inertiaOrientation defines the transformation matrix between both (T_LK)
def steinerRule(mass, inertia, com, inertiaOrientation = None):
  T_LK = numpy.array(inertiaOrientation) if inertiaOrientation is not None else numpy.eye(3)
  return  T_LK @ numpy.array(inertia) @ T_LK.T + mass * tilde(numpy.array(com)).T @ tilde(numpy.array(com))



# Sum up mass values.
# massValue is a dictionary with the keys
# - "mass":               the mass of body to add
# - "inertia":            the inertia tensor of the body to add; the inertia must be given around its center of mass "com" and with respect to a local coordinate system L
# - "com":                the center of mass of the body to add; the com must be given with respect to a local coordinate system L
# - "inertiaOrientation": optional, if given, "inertia" is not with respect to the local system L but to a system K and inertiaOrientation defines the transformation matrix between both (T_LK)
# Returned is a dictionary with the same keys being the summed up mass values of all arguments where
# - "inertia":            is given around the summed up center of mass and with respect to the local coordinate system L.
# - "com":                is with respect to the local cooridnate system L.
# - "inertiaOrientation": is not provided since "inertia" is always returned with respect to the local coordinate system L (T_LK = eye).
def sumMassValues(*massValue):
  sum_m = 0
  sum_inertia_L = numpy.zeros((3,3))
  sum_mcom = numpy.zeros(3)
  for mv in massValue:
    sum_m += mv["mass"]
    sum_inertia_L += steinerRule(mv["mass"], mv["inertia"], mv["com"], mv.get("inertiaOrientation", numpy.eye(3))) # positive mass -> move from com to none-com
    sum_mcom += mv["mass"] * numpy.array(mv["com"])
  sum_com = sum_mcom / sum_m
  sum_inertia_C = steinerRule(-sum_m, sum_inertia_L, sum_com) # negative mass -> move from none-com to com
  return { "mass": sum_m, "inertia": sum_inertia_C, "com": sum_com}



# return the HSV color of the given RGB values (0-1)
def rgbColor(*argv):
  import colorsys
  if len(argv)==3:
    red=argv[0]
    green=argv[1]
    blue=argv[2]
  elif len(argv)==1 and len(argv[0])==3:
    red=argv[0][0]
    green=argv[0][1]
    blue=argv[0][2]
  else:
    raise RuntimeError('Must be called with a three scalar arguments or one vector argument of length 3.')
  return numpy.array(colorsys.rgb_to_hsv(red, green, blue))



# return the HSV color of the given CSS color name which can also be a HEX color (#XXXXXX)
# see https://www.w3.org/wiki/CSS/Properties/color/keywords
def namedColor(name):
  if namedColor.nameToHEX is None:
    namedColor.nameToHEX={
      "black": "#000000",
      "silver": "#c0c0c0",
      "gray": "#808080",
      "white": "#ffffff",
      "maroon": "#800000",
      "red": "#ff0000",
      "purple": "#800080",
      "fuchsia": "#ff00ff",
      "green": "#008000",
      "lime": "#00ff00",
      "olive": "#808000",
      "yellow": "#ffff00",
      "navy": "#000080",
      "blue": "#0000ff",
      "teal": "#008080",
      "aqua": "#00ffff",

      "aliceblue": "#f0f8ff",
      "antiquewhite": "#faebd7",
      "aqua": "#00ffff",
      "aquamarine": "#7fffd4",
      "azure": "#f0ffff",
      "beige": "#f5f5dc",
      "bisque": "#ffe4c4",
      "black": "#000000",
      "blanchedalmond": "#ffebcd",
      "blue": "#0000ff",
      "blueviolet": "#8a2be2",
      "brown": "#a52a2a",
      "burlywood": "#deb887",
      "cadetblue": "#5f9ea0",
      "chartreuse": "#7fff00",
      "chocolate": "#d2691e",
      "coral": "#ff7f50",
      "cornflowerblue": "#6495ed",
      "cornsilk": "#fff8dc",
      "crimson": "#dc143c",
      "cyan": "#00ffff",
      "darkblue": "#00008b",
      "darkcyan": "#008b8b",
      "darkgoldenrod": "#b8860b",
      "darkgray": "#a9a9a9",
      "darkgreen": "#006400",
      "darkgrey": "#a9a9a9",
      "darkkhaki": "#bdb76b",
      "darkmagenta": "#8b008b",
      "darkolivegreen": "#556b2f",
      "darkorange": "#ff8c00",
      "darkorchid": "#9932cc",
      "darkred": "#8b0000",
      "darksalmon": "#e9967a",
      "darkseagreen": "#8fbc8f",
      "darkslateblue": "#483d8b",
      "darkslategray": "#2f4f4f",
      "darkslategrey": "#2f4f4f",
      "darkturquoise": "#00ced1",
      "darkviolet": "#9400d3",
      "deeppink": "#ff1493",
      "deepskyblue": "#00bfff",
      "dimgray": "#696969",
      "dimgrey": "#696969",
      "dodgerblue": "#1e90ff",
      "firebrick": "#b22222",
      "floralwhite": "#fffaf0",
      "forestgreen": "#228b22",
      "fuchsia": "#ff00ff",
      "gainsboro": "#dcdcdc",
      "ghostwhite": "#f8f8ff",
      "gold": "#ffd700",
      "goldenrod": "#daa520",
      "gray": "#808080",
      "green": "#008000",
      "greenyellow": "#adff2f",
      "grey": "#808080",
      "honeydew": "#f0fff0",
      "hotpink": "#ff69b4",
      "indianred": "#cd5c5c",
      "indigo": "#4b0082",
      "ivory": "#fffff0",
      "khaki": "#f0e68c",
      "lavender": "#e6e6fa",
      "lavenderblush": "#fff0f5",
      "lawngreen": "#7cfc00",
      "lemonchiffon": "#fffacd",
      "lightblue": "#add8e6",
      "lightcoral": "#f08080",
      "lightcyan": "#e0ffff",
      "lightgoldenrodyellow": "#fafad2",
      "lightgray": "#d3d3d3",
      "lightgreen": "#90ee90",
      "lightgrey": "#d3d3d3",
      "lightpink": "#ffb6c1",
      "lightsalmon": "#ffa07a",
      "lightseagreen": "#20b2aa",
      "lightskyblue": "#87cefa",
      "lightslategray": "#778899",
      "lightslategrey": "#778899",
      "lightsteelblue": "#b0c4de",
      "lightyellow": "#ffffe0",
      "lime": "#00ff00",
      "limegreen": "#32cd32",
      "linen": "#faf0e6",
      "magenta": "#ff00ff",
      "maroon": "#800000",
      "mediumaquamarine": "#66cdaa",
      "mediumblue": "#0000cd",
      "mediumorchid": "#ba55d3",
      "mediumpurple": "#9370db",
      "mediumseagreen": "#3cb371",
      "mediumslateblue": "#7b68ee",
      "mediumspringgreen": "#00fa9a",
      "mediumturquoise": "#48d1cc",
      "mediumvioletred": "#c71585",
      "midnightblue": "#191970",
      "mintcream": "#f5fffa",
      "mistyrose": "#ffe4e1",
      "moccasin": "#ffe4b5",
      "navajowhite": "#ffdead",
      "navy": "#000080",
      "oldlace": "#fdf5e6",
      "olive": "#808000",
      "olivedrab": "#6b8e23",
      "orange": "#ffa500",
      "orangered": "#ff4500",
      "orchid": "#da70d6",
      "palegoldenrod": "#eee8aa",
      "palegreen": "#98fb98",
      "paleturquoise": "#afeeee",
      "palevioletred": "#db7093",
      "papayawhip": "#ffefd5",
      "peachpuff": "#ffdab9",
      "peru": "#cd853f",
      "pink": "#ffc0cb",
      "plum": "#dda0dd",
      "powderblue": "#b0e0e6",
      "purple": "#800080",
      "red": "#ff0000",
      "rosybrown": "#bc8f8f",
      "royalblue": "#4169e1",
      "saddlebrown": "#8b4513",
      "salmon": "#fa8072",
      "sandybrown": "#f4a460",
      "seagreen": "#2e8b57",
      "seashell": "#fff5ee",
      "sienna": "#a0522d",
      "silver": "#c0c0c0",
      "skyblue": "#87ceeb",
      "slateblue": "#6a5acd",
      "slategray": "#708090",
      "slategrey": "#708090",
      "snow": "#fffafa",
      "springgreen": "#00ff7f",
      "steelblue": "#4682b4",
      "tan": "#d2b48c",
      "teal": "#008080",
      "thistle": "#d8bfd8",
      "tomato": "#ff6347",
      "turquoise": "#40e0d0",
      "violet": "#ee82ee",
      "wheat": "#f5deb3",
      "white": "#ffffff",
      "whitesmoke": "#f5f5f5",
      "yellow": "#ffff00",
      "yellowgreen": "#9acd32",
    }
  hexColor = namedColor.nameToHEX.get(name.lower(), name)
  if hexColor[0]!="#" or len(hexColor)!=7:
    raise RuntimeError(f"Invalid color name: {name}")
  return rgbColor(int(hexColor[1:3], 16)/255, int(hexColor[3:5], 16)/255, int(hexColor[5:7], 16)/255)
namedColor.nameToHEX=None



# a simply dictionary on module level to store data.
# the lifetime of data is the lifetime of the python evaluator.
# This can be used to store any data and access it later on (but take care about memory usage when large data is stored)
staticData=dict()
