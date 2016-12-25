import numpy
import casadi



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
  elif len(argv)==1 and ((type(argv[0]).__name__!="SX" and len(argv[0])==3) or \
                         (type(argv[0]).__name__=="SX" and argv[0].shape==(3,1))):
    alpha=argv[0][0];
    beta=argv[0][1];
    gamma=argv[0][2];
  else:
    raise RuntimError('Must be called with a three scalar arguments or one vector argument of length 3.')

  return numpy.array([[casadi.cos(beta)*casadi.cos(gamma),
                     -casadi.cos(beta)*casadi.sin(gamma),
                     casadi.sin(beta)],
                     [casadi.cos(alpha)*casadi.sin(gamma)+casadi.sin(alpha)*casadi.sin(beta)*casadi.cos(gamma),
                     casadi.cos(alpha)*casadi.cos(gamma)-casadi.sin(alpha)*casadi.sin(beta)*casadi.sin(gamma),
                     -casadi.sin(alpha)*casadi.cos(beta)],
                     [casadi.sin(alpha)*casadi.sin(gamma)-casadi.cos(alpha)*casadi.sin(beta)*casadi.cos(gamma),
                     casadi.cos(alpha)*casadi.sin(beta)*casadi.sin(gamma)+casadi.sin(alpha)*casadi.cos(gamma),
                     casadi.cos(alpha)*casadi.cos(beta)]])



def euler(*argv):
  if len(argv)==3:
    PHI=argv[0]
    theta=argv[1]
    phi=argv[2]
  elif len(argv)==1 and ((type(argv[0]).__name__!="SX" and len(argv[0])==3) or \
                         (type(argv[0]).__name__=="SX" and argv[0].shape==(3,1))):
    PHI=argv[0][0]
    theta=argv[0][1]
    phi=argv[0][2]
  else:
    raise RuntimeError('Must be called with a three scalar arguments or one vector argument of length 3.')

  return numpy.array([[casadi.cos(phi)*casadi.cos(PHI)-casadi.sin(phi)*casadi.cos(theta)*casadi.sin(PHI),
                     -casadi.cos(phi)*casadi.cos(theta)*casadi.sin(PHI)-casadi.sin(phi)*casadi.cos(PHI),
                     casadi.sin(theta)*casadi.sin(PHI)],
                     [casadi.cos(phi)*casadi.sin(PHI)+casadi.sin(phi)*casadi.cos(theta)*casadi.cos(PHI),
                     casadi.cos(phi)*casadi.cos(theta)*casadi.cos(PHI)-casadi.sin(phi)*casadi.sin(PHI),
                     -casadi.sin(theta)*casadi.cos(PHI)],
                     [casadi.sin(phi)*casadi.sin(theta),
                     casadi.cos(phi)*casadi.sin(theta),
                     casadi.cos(theta)]])



def rotateAboutX(phi):
  return numpy.array([[1,0,0],
                     [0,casadi.cos(phi),-casadi.sin(phi)],
                     [0,casadi.sin(phi),casadi.cos(phi)]])



def rotateAboutY(phi):
  return numpy.array([[casadi.cos(phi),0,casadi.sin(phi)],
                     [0,1,0],
                     [-casadi.sin(phi),0,casadi.cos(phi)]])



def rotateAboutZ(phi):
  return numpy.array([[casadi.cos(phi),-casadi.sin(phi),0],
                     [casadi.sin(phi),casadi.cos(phi),0],
                     [0,0,1]])



def tilde(x):
                
  if (type(x).__name__=="SX" and x.shape==(3,1)) or \
     (type(x).__name__!="SX" and len(x)==3 and not hasattr(x[0], '__len__')):

    return numpy.array([[    0, -x[2],  x[1]], \
                        [ x[2],     0, -x[0]], \
                        [-x[1],  x[0],     0]])

  elif (type(x).__name__=="SX" and x.shape==(3,3)) or (len(x)==3 and len(x[0])==3):

    if type(x).__name__!="SX":
      abserr=1e-7;
      if abs(x[0][0])>abserr or abs(x[1][1])>abserr or abs(x[2][2])>abserr or \
         abs(x[0][1]+x[1][0])>abserr or abs(x[0][2]+x[2][0])>abserr or abs(x[1][2]+x[2][1])>abserr:
        raise RuntimeError('Must be s skew symmetric matrix.')
      return numpy.array([x[2][1], x[0][2], x[1][0]])
    else:
      return numpy.array([x[2,1], x[0,2], x[1,0]])

  else:
    raise RuntimeError('Must be called with with a 3x3 matrix or a column vector of length 3.')
