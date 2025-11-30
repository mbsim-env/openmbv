"""MBXMLUtils internal helper functions"""



import io
import os



class _CppOStream(io.IOBase):
  def __init__(self, cppOStreamPtr_):
    self.cppOStreamPtr=cppOStreamPtr_
  def write(self, data):
    import ctypes
    import mbxmlutils
    mbxmlutils._getPyEvalDLL().mbxmlutils_output.argtypes=[ctypes.c_void_p, ctypes.c_char_p]
    mbxmlutils._getPyEvalDLL().mbxmlutils_output.restype=None
    mbxmlutils._getPyEvalDLL().mbxmlutils_output(self.cppOStreamPtr, data.encode("utf-8"))



# INTERNAL FUNCTION
# serialize a symbolic expression to the fmatvec format
def _serializeFunction(x):
  import numpy
  # serialize a matrix or vector (we allow sympy matrices, numpy array and python lists)
  if x.__class__.__name__=="MutableDenseMatrix" or x.__class__.__name__=="ImmutableDenseMatrix" or \
     type(x)==numpy.ndarray:
    s=["["]
    if len(x.shape)==2:
      (rows, cols)=x.shape
      for r in range(0, rows):
        for c in range(0, cols):
          s.append(_serializeFunction(x[r,c]))
          if c<cols-1:
            s.append(", ")
        if r<rows-1:
          s.append("; ")
    elif len(x.shape)==1:
      rows=x.shape[0]
      for r in range(0, rows):
        s.append(_serializeFunction(x[r]))
        if r<rows-1:
          s.append("; ")
    else:
      raise RuntimeError("Internal error: Unknonw shape")
    s.append("]")
    return "".join(s)
  elif type(x)==list:
    s=["["]
    if len(x)>0 and type(x[0])==list:
      rows=len(x)
      for r in range(0, rows):
        cols=len(x[r])
        for c in range(0, cols):
          s.append(_serializeFunction(x[r][c]))
          if c<cols-1:
            s.append(", ")
        if r<rows-1:
          s.append("; ")
    elif (len(x)>0 and type(x[0])!=list) or len(x)==0:
      rows=len(x)
      for r in range(0, rows):
        s.append(_serializeFunction(x[r]))
        if r<rows-1:
          s.append("; ")
    else:
      raise RuntimeError("Internal error: Unknonw shape")
    s.append("]")
    return "".join(s)
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
      if x.func.__name__=="Dummy" or x.func.__name__=="Symbol":
        import uuid
        # create a unique UUID for the variable (map the object x to a UUID)
        uid=_serializeFunction.indepMap.setdefault(x, uuid.uuid4())
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

          def boolVertexToGt0Func(c):
            if list(map(lambda x: int(x), sympy.__version__.split(".")[0:2])) >= [1,6]:
              _sympyRelational=sympy.core.relational
            else:
              _sympyRelational=sympy.relational
          
            if isinstance(c, _sympyRelational.GreaterThan) or isinstance(c, _sympyRelational.StrictGreaterThan):
              return c.lhs-c.rhs
            elif isinstance(c, _sympyRelational.LessThan) or isinstance(c, _sympyRelational.StrictLessThan):
              return c.rhs-c.lhs
            elif c==True:
              return 1
            elif c==False:
              return -1
            elif isinstance(c, sympy.logic.boolalg.And):
              return sympy.Min(*map(lambda x: boolVertexToGt0Func(x), c.args))
            elif isinstance(c, sympy.logic.boolalg.Or):
              return sympy.Max(*map(lambda x: boolVertexToGt0Func(x), c.args))
            else:
              raise RuntimeError("Unknown relation in Piecewise: "+str(c))
          cGt0Func=boolVertexToGt0Func(c)
          return "".join(["condition(",serializeVertex(cGt0Func),",",serializeVertex(gt),",",serializeVertex(le),")"])
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
      s=[opStr[0],"("]
      first=True
      for op in x.args[0:opStr[1]]:
        if not first: s.append(",")
        s.append(serializeVertex(op))
        first=False
      s.append(")")
      return "".join(s)
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
