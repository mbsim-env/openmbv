#include <config.h>

// python includes
#include <Python.h> // due to some bugs in python 3.2 we need to include this first
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// normal includes
#include "pyeval.h"

// other includes
#include <cfenv>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/join.hpp>
#include "mbxmlutils/eval_static.h"
#include <memory>
#include <regex>

using namespace std;
using namespace boost::filesystem;
using namespace xercesc;
using namespace PythonCpp;

namespace {

inline PyO C(const MBXMLUtils::Eval::Value &value) {
  return *static_pointer_cast<PyO>(value);
}

inline MBXMLUtils::Eval::Value C(const PyO &value) {
  return make_shared<PyO>(value);
}

bool is_scalar_double(const MBXMLUtils::Eval::Value &value);
bool is_vector_double(const MBXMLUtils::Eval::Value &value, PyArrayObject** a=nullptr, int *type=nullptr);
bool is_vector_vector_double(const MBXMLUtils::Eval::Value &value, PyArrayObject **a=nullptr, int *type=nullptr);

double arrayScalarGetDouble(PyObject *o, bool *error=nullptr);
double arrayGetDouble(PyArrayObject *a, int type, int r, int c=-1);

}

namespace MBXMLUtils {

MBXMLUTILS_EVAL_REGISTER(PyEval)

class PyOOnDemand {
  public:
    PyOOnDemand() = default;
    PyOOnDemand(const function<PyO()> &create_) : create(create_) {}
    PyO operator()() {
      if(!o)
        o=create();
      return o;
    }
    operator bool() const {
      return o;
    }
  private:
    function<PyO()> create;
    PyO o;
};

// Helper class to init/deinit Python on library load/unload (unload=program end)
class PyInit {
  public:
    PyInit();
    ~PyInit();
    PyOOnDemand mbxmlutils;
    map<string, PyO> functionValue;
    PyOOnDemand asarray;
    PyOOnDemand zeros;
    PyOOnDemand sympy;
    PyOOnDemand dummy;
    PyOOnDemand matrix;
    PyOOnDemand serializeFunction;
};

PyInit::PyInit() {
  try {
    // init python
    initializePython(Eval::installPath/"bin"/"mbxmlutilspp", PYTHON_VERSION, {
      // append mbxmlutils module to the python path (the basic Python module for the pyeval)
      Eval::installPath/"share"/"mbxmlutils"/"python",
      // append the installation/bin dir to the python path (SWIG generated python modules (e.g. OpenMBV.py) are located there)
      Eval::installPath/"bin",
      // prepand the installation/../mbsim-env-python-site-packages dir to the python path (Python pip of mbsim-env is configured to install user defined python packages there)
      Eval::installPath.parent_path()/"mbsim-env-python-site-packages",
    }, {
      Eval::installPath,
      boost::filesystem::path(PYTHON_PREFIX),
    });

    // init numpy
    CALLPY(_import_array);

    sympy=PyOOnDemand([](){ return CALLPY(PyImport_ImportModule, "sympy"); });

    mbxmlutils=PyOOnDemand([](){ return CALLPY(PyImport_ImportModule, "mbxmlutils"); });

    asarray=PyOOnDemand([](){
      PyO numpy(CALLPY(PyImport_ImportModule, "numpy"));
      return CALLPY(PyObject_GetAttrString, numpy, "asarray");
    });

    zeros=PyOOnDemand([](){
      PyO numpy(CALLPY(PyImport_ImportModule, "numpy"));
      return CALLPY(PyObject_GetAttrString, numpy, "zeros");
    });

    dummy=PyOOnDemand([this](){ return CALLPY(PyObject_GetAttrString, sympy(), "Dummy"); });
    matrix=PyOOnDemand([this](){ return CALLPY(PyObject_GetAttrString, sympy(), "Matrix"); });
    serializeFunction=PyOOnDemand([this](){ return CALLPY(PyObject_GetAttrString, mbxmlutils(), "_serializeFunction"); });
  }
  // print error and rethrow. (The exception may not be catched since this is called in pre-main)
  catch(const exception& ex) {
    fmatvec::Atom::msgStatic(fmatvec::Atom::Error)<<"Exception during Python initialization:"<<endl<<ex.what()<<endl;
    throw;
  }
  catch(...) {
    fmatvec::Atom::msgStatic(fmatvec::Atom::Error)<<"Unknown exception during Python initialization."<<endl;
    throw;
  }
}

PyInit::~PyInit() {
  try {
    // clear all Python object before deinit
    if(serializeFunction) serializeFunction().reset();
    if(matrix) matrix().reset();
    if(dummy) dummy().reset();
    if(sympy) sympy().reset();
    if(asarray) asarray().reset();
    if(zeros) zeros().reset();
    functionValue.clear();
    if(mbxmlutils) mbxmlutils().reset();
  }
  // print error and rethrow. (The exception may not be catched since this is called in pre-main)
  catch(const exception& ex) {
    fmatvec::Atom::msgStatic(fmatvec::Atom::Error)<<"Exception during Python deinitialization:"<<endl<<ex.what()<<endl
      <<"Continuing but undefined behaviour may occur."<<endl;
  }
  catch(...) {
    fmatvec::Atom::msgStatic(fmatvec::Atom::Error)<<"Unknown exception during Python deinitialization."<<endl
      <<"Continuing but undefined behaviour may occur."<<endl;
  }
}

// we cannot call the ctor of PyInit here as a global variable!
// Some python modules do not link against libpython but rely on the fact the libpyhton is already
// loaded into the global symbol space. Hence, this library must be loaded using RTDL_GLOBAL and
// the initialization cannot be done in global ctor code, nor in using the _init function of ld nor
// using the GNU __attribute__((constructor)) because this does not work! But calling a init function
// after the so is fully loaded does work. Hence, we define the MBXMLUtils_SharedLibrary_init function
// here which is executed by SharedLibrary::load after the lib is loaded.
// On Windows this is not required, but does not hurt.
unique_ptr<PyInit> pyInit; // init Python on library load and deinit on library unload = program end

extern "C"
int MBXMLUtils_SharedLibrary_init() {
  try {
    pyInit=std::make_unique<PyInit>();
  }
  catch(...) {
    return 1;
  }
  return 0;
}

PyEval::PyEval(vector<path> *dependencies_) : Eval(dependencies_) {
  currentImport=make_shared<PyO>(CALLPY(PyDict_New));
}

PyEval::~PyEval() = default;

Eval::Value PyEval::createFunctionIndep(int dim) const {
  PyO indep;
  if(dim==0)
    indep=CALLPY(PyObject_CallObject, pyInit->dummy(), CALLPY(PyTuple_New, 0));
  else {
    PyO list(CALLPY(PyList_New, dim));
    for(size_t i=0; i<dim; ++i)
      CALLPY(PyList_SetItem, list, i, CALLPYB(PyObject_CallObject, pyInit->dummy(), CALLPY(PyTuple_New, 0)));
    PyO arg(CALLPY(PyTuple_New, 1));
    CALLPY(PyTuple_SetItem, arg, 0, list.incRef());
    indep=CALLPY(PyObject_CallObject, pyInit->matrix(), arg);
  }
  return make_shared<PyO>(indep);
}

namespace {
  string fixPythonIndentation(const string &str, const DOMElement *e) {
    // fix python indentation
    vector<string> lines;
    boost::split(lines, str, boost::is_any_of("\n")); // split to a vector of lines
    size_t indent=string::npos;
    size_t lineNr=0;
    for(auto it=lines.begin(); it!=lines.end(); ++it, ++lineNr) {
      size_t pos=it->find_first_not_of(' '); // get first none space character
      if(pos==string::npos) continue; // not found -> pure empty line -> do not modify
      if(pos!=string::npos && (*it)[pos]=='#') continue; // found and first char is '#' -> pure comment line -> do not modify
      // now we have a line with a python statement
      if(indent==string::npos) indent=pos; // at the first python statement line use the current indent as indent for all others
      if(it->substr(0, indent)!=string(indent, ' ')) // check if line starts with at least indent spaces ...
        // ... if not its an indentation error
        throw DOMEvalException("Unexpected indentation at line "+fmatvec::toString(lineNr)+": "+str, e);
      *it=it->substr(indent); // remove the first indent spaces from the line
    }
    return boost::join(lines, "\n"); // join the lines to a single string
  }
}

void PyEval::addImport(const string &code, const DOMElement *e) {
  // python globals (fill with builtins)
  PyO globalsLocals(CALLPY(PyDict_New));
  CALLPY(PyDict_SetItemString, globalsLocals, "__builtins__", CALLPYB(PyEval_GetBuiltins));
  // python globals (fill with current parameters)
  for(auto i=currentParam.begin(); i!=currentParam.end(); i++)
    CALLPY(PyDict_SetItemString, globalsLocals, i->first, C(i->second));

  // get current python sys.path
  auto getSysPath=[](){
    set<string> sysPath;
    PyO path=CALLPYB(PySys_GetObject, const_cast<char*>("path"));
    int size=CALLPY(PyList_Size, path);
    for(int i=0; i<size; ++i) {
      PyO str(CALLPYB(PyList_GetItem, path, i));
      sysPath.insert(CALLPY(PyUnicode_AsUTF8, str));
    }
    return sysPath;
  };
  set<string> oldPath;
  if(dependencies)
    oldPath=getSysPath();

  // evaluate as statement with current path set
  {
    // restore current dir on exit and change current dir
    PreserveCurrentDir preserveDir;
    path chdir=E(e)->getOriginalFilename().parent_path();
    if(!chdir.empty())
      current_path(chdir);

    PyCompilerFlags flags;
    flags.cf_flags=CO_FUTURE_DIVISION; // we evaluate the code in python 3 mode (future python 2 mode)
    try {
      auto codetrim=fixPythonIndentation(code, e);
      mbxmlutilsStaticDependencies.clear();
      originalFilename=E(e)->getOriginalFilename();
      CALLPY(PyRun_StringFlags, codetrim, Py_file_input, globalsLocals, globalsLocals, &flags);
      addStaticDependencies(e);
    }
    catch(const exception& ex) { // on failure -> report error
      throw DOMEvalException(string(ex.what())+"Unable to evaluate Python code:\n"+code, e);
    }
  }

  // get all locals and add to currentImport
  CALLPY(PyDict_Merge, *static_pointer_cast<PyO>(currentImport), globalsLocals, true);

  if(dependencies) {
    // get current python sys.path
    set<string> newPath=getSysPath();
    // add py-files in added sys.path to dependencies
    for(auto &np: newPath) {
      // skip path already existing in oldPath
      if(oldPath.find(np)!=oldPath.end())
        continue;
      path dir=E(e)->convertPath(np);
      for(directory_iterator it=directory_iterator(dir); it!=directory_iterator(); it++)
        if(it->path().extension()==".py")
          dependencies->push_back(it->path());
    }
  }
}

bool PyEval::valueIsOfType(const Value &value, ValueType type) const {
  PyO v(C(value));
  switch(type) {
    case ScalarType: return ::is_scalar_double(value);
    case VectorType: return ::is_vector_double(value);
    case MatrixType: return ::is_vector_vector_double(value);
    case StringType: return CALLPY(PyUnicode_Check, v);
    case FunctionType: return PyTuple_Check(v.get());
  }
  throw runtime_error("Internal error: Unknwon ValueType.");
}

path relative(const path& abs, const path& relTo) {
  size_t dist=distance(relTo.begin(), relTo.end());
  path::iterator dstIt=abs.begin();
  for(int i=0; i<dist; ++i) ++dstIt;
  path ret;
  for(; dstIt!=abs.end(); ++dstIt)
    ret/=*dstIt;
  return ret;
}

path replace_extension(path p, const path& newExt=path()) {
  return p.replace_extension(newExt);
}

map<path, pair<path, bool> >& PyEval::requiredFiles() const {
  static map<path, pair<path, bool> > files;
  if(!files.empty())
    return files;

  path PYTHONDST(PYTHON_SUBDIR);

  fmatvec::Atom::msgStatic(fmatvec::Atom::Info)<<"Generate file list for MBXMLUtils py-files."<<endl;
  for(auto srcIt=directory_iterator(installPath/"share"/"mbxmlutils"/"python"); srcIt!=directory_iterator(); ++srcIt) {
    if(is_directory(*srcIt)) // skip directories
      continue;
    files[srcIt->path()]=make_pair(path("share")/"mbxmlutils"/"python", false);
  }

  fmatvec::Atom::msgStatic(fmatvec::Atom::Info)<<"Generate file list for Python files."<<endl;
  path PYTHONSRC(PYTHON_LIBDIR);
  if(exists(installPath/PYTHON_SUBDIR/"site-packages"))
    PYTHONSRC=installPath/PYTHON_SUBDIR;
  for(auto srcIt=recursive_directory_iterator(PYTHONSRC); srcIt!=recursive_directory_iterator(); ++srcIt) {
    if(is_directory(*srcIt)) // skip directories
      continue;
    path subDir=MBXMLUtils::relative(*srcIt, PYTHONSRC).parent_path();
    if((*subDir.begin()=="site-packages" || *subDir.begin()=="dist-packages") &&
      *(++subDir.begin())!="numpy" && *(++subDir.begin())!="sympy" && *(++subDir.begin())!="mpmath") // skip site-packages dir but not numpy and sympy/mpmath
      continue;
    if(*subDir.begin()=="config") // skip config dir
      continue;
    bool bin=false;
    const regex so(".*\\.so(\\..*)*");
    if(srcIt->path().extension()==".dll" || srcIt->path().extension()==".pyd" ||
       regex_match(srcIt->path().filename().string(), so))
      bin=true;
    files[*srcIt]=make_pair(PYTHONDST/subDir, bin);
  }
#if _WIN32
  // on Windows include the PYTHONSRC/../DLLs directory, if existing
  if(exists(PYTHONSRC/".."/"DLLs"))
    for(auto srcIt=directory_iterator(PYTHONSRC/".."/"DLLs"); srcIt!=directory_iterator(); ++srcIt)
      files[srcIt->path()]=make_pair("DLLs", false); // just copy these files, dependencies are handled by python
#endif

  return files;
}

Eval::Value PyEval::callFunction(const string &name, const vector<Value>& args) const {
  pair<map<string, PyO>::iterator, bool> f=pyInit->functionValue.insert(make_pair(name, PyO()));
  if(f.second)
    f.first->second=CALLPYB(PyDict_GetItemString, CALLPYB(PyModule_GetDict, pyInit->mbxmlutils()), name);
  PyO pyargs(CALLPY(PyTuple_New, args.size()));
  int idx=0;
  for(auto it=args.begin(); it!=args.end(); ++it, ++idx) {
    PyO v(C(*it));
    CALLPY(PyTuple_SetItem, pyargs, idx, v.incRef()); // PyTuple_SetItem steals a reference of v
  }
  return C(CALLPY(PyObject_CallObject, f.first->second, pyargs));
}

Eval::Value PyEval::fullStringToValue(const string &str, const DOMElement *e, bool skipRet) const {
  string strtrim=str;
  boost::trim(strtrim);

  // check some common string to avoid time consiming evaluation
  // check true and false
  if(strtrim=="True") return C(CALLPY(PyBool_FromLong, 1));
  if(strtrim=="False") return C(CALLPY(PyBool_FromLong, 0));
  // check for integer and floating point values
  double d;
  char *end;
  d=strtod(strtrim.c_str(), &end);
  if(string(end).empty()) {
    int i;
    if(tryDouble2Int(d, i))
      return C(CALLPY(PyLong_FromLong, i));
    return C(CALLPY(PyFloat_FromDouble, d));
  }
  // no common string detected -> evaluate using python now

  // restore current dir on exit and change current dir
  PreserveCurrentDir preserveDir;
  if(e) {
    path chdir=E(e)->getOriginalFilename().parent_path();
    if(!chdir.empty())
      current_path(chdir);
  }

  // python globals (fill with builtins)
  PyO globalsLocals(CALLPY(PyDict_New));
  CALLPY(PyDict_SetItemString, globalsLocals, "__builtins__", CALLPYB(PyEval_GetBuiltins));
  // python globals (fill with imports)
  CALLPY(PyDict_Merge, globalsLocals, *static_pointer_cast<PyO>(currentImport), true);
  // python globals (fill with current parameters)
  for(auto i=currentParam.begin(); i!=currentParam.end(); i++)
    CALLPY(PyDict_SetItemString, globalsLocals, i->first, C(i->second));

  PyO ret;
  PyCompilerFlags flags;
  flags.cf_flags=CO_FUTURE_DIVISION; // we evaluate the code in python 3 mode (future python 2 mode)

  // evaluate as expression (using the trimmed str) and save result in ret
  mbxmlutilsStaticDependencies.clear();
  if(e)
    originalFilename=E(e)->getOriginalFilename();
  else
    originalFilename.clear();
  PyObject* pyo=PyRun_StringFlags(strtrim.c_str(), Py_eval_input, globalsLocals.get(), globalsLocals.get(), &flags);
  // clear the python exception in case of errors (done by creating a dummy PythonException object)
  if(PyErr_Occurred())
    PythonException dummy("", 0);
  if(pyo) { // on success ...
    ret=PyO(pyo);
    addStaticDependencies(e);
  }
  else { // on failure ...
    try {
      // ... evaluate as statement

      strtrim=fixPythonIndentation(str, e);

      // evaluate as statement
      mbxmlutilsStaticDependencies.clear();
      if(e)
        originalFilename=E(e)->getOriginalFilename();
      else
        originalFilename.clear();
      CALLPY(PyRun_StringFlags, strtrim, Py_file_input, globalsLocals, globalsLocals, &flags);
      addStaticDependencies(e);
    }
    catch(const exception& ex) { // on failure -> report error
      throw DOMEvalException(string(ex.what())+"Unable to evaluate Python code:\n"+str, e);
    }
    if(!skipRet) {
      try {
        // get 'ret' variable from statement
        ret=CALLPYB(PyDict_GetItemString, globalsLocals, "ret");
      }
      catch(const exception&) {
        // 'ret' variable not found or invalid expression
        throw DOMEvalException("Invalid expression or statement does not define the 'ret' variable in expression:\n"+str, e);
      }
    }
  }
  if(skipRet)
    return {};
  // convert a list of scalars / list of lists of scalars to a numpy 1D / 2D array, respectively
  // * a list of len==0 is interpreted as a R^0 vector [shape==(0,)]
  if(CALLPY(PyList_Check, ret) && CALLPY(PyList_Size, ret)==0) {
    PyO args(CALLPY(PyTuple_New, 1));
    CALLPY(PyTuple_SetItem, args, 0, CALLPY(PyLong_FromLong, 0).incRef()); // PyTuple_SetItem steals a reference of ret
    return C(CALLPY(PyObject_CallObject, pyInit->zeros(), args));
  }
  // * note that in python you cannot create R^0xC matrix from a python list
  // * a list with at least one element is converted using numpy.asarray (if the first element is a scalar, a empty list or a list of doubles)
  if(CALLPY(PyList_Check, ret) && CALLPY(PyList_Size, ret)>0) {
    PyO ret0(CALLPYB(PyList_GetItem, ret, 0));
    if(is_scalar_double(C(ret0)) ||
       (CALLPY(PyList_Check, ret0) && CALLPY(PyList_Size, ret0)==0) ||
       (CALLPY(PyList_Check, ret0) && CALLPY(PyList_Size, ret0)>0 && is_scalar_double(C(CALLPYB(PyList_GetItem, ret0, 0))))) {
      PyO args(CALLPY(PyTuple_New, 1));
      CALLPY(PyTuple_SetItem, args, 0, ret.incRef()); // PyTuple_SetItem steals a reference of ret
      return C(CALLPY(PyObject_CallObject, pyInit->asarray(), args));
    }
  }
  // return result
  return C(ret);
}

double PyEval::cast_double(const Value &value) const {
  PyO v(C(value));
  if(CALLPY(PyFloat_Check, v))
    return CALLPY(PyFloat_AsDouble, v);
  if(CALLPY(PyLong_Check, v))
    return CALLPY(PyLong_AsLong, v);
  if(PyArray_CheckScalar(v.get()))
    return arrayScalarGetDouble(v.get());
  throw runtime_error("Cannot cast this value to double.");
}

vector<double> PyEval::cast_vector_double(const Value &value) const {
  PyArrayObject* a;
  int type;
  if(!is_vector_double(value, &a, &type))
    throw runtime_error("Value is not of type vector.");
  
  npy_intp *dims=PyArray_SHAPE(a);
  vector<double> ret(dims[0]);
  for(size_t i=0; i<dims[0]; ++i)
    ret[i]=arrayGetDouble(a, type, i);
  return ret;
}

vector<vector<double> >PyEval::cast_vector_vector_double(const Value &value) const {
  PyArrayObject* a;
  int type;
  if(!is_vector_vector_double(value, &a, &type))
    throw runtime_error("Value is not of type matrix.");

  npy_intp *dims=PyArray_SHAPE(a);
  vector<vector<double> > ret(dims[0], vector<double>(dims[1]));
  for(size_t r=0; r<dims[0]; ++r)
    for(size_t c=0; c<dims[1]; ++c)
      ret[r][c]=arrayGetDouble(a, type, r, c);
  return ret;
}

string PyEval::cast_string(const Value &value) const {
  return CALLPY(PyUnicode_AsUTF8, C(value));
}

Eval::Value PyEval::create_double(const double& v) const {
  int i;
  if(tryDouble2Int(v, i))
    return C(CALLPY(PyLong_FromLong, i));
  return C(CALLPY(PyFloat_FromDouble, v));
}

Eval::Value PyEval::create_vector_double(const vector<double>& v) const {
  npy_intp dims[1];
  dims[0]=v.size();
  PyO ret(PyArray_SimpleNew(1, dims, NPY_DOUBLE));
  copy(v.begin(), v.end(), static_cast<npy_double*>(PyArray_GETPTR1(reinterpret_cast<PyArrayObject*>(ret.get()), 0)));
  return C(ret);
}

Eval::Value PyEval::create_vector_vector_double(const vector<vector<double> >& v) const {
  npy_intp dims[2];
  dims[0]=v.size();
  dims[1]=v[0].size();
  PyO ret(PyArray_SimpleNew(2, dims, NPY_DOUBLE));
  int r=0;
  for(auto it=v.begin(); it!=v.end(); ++it, ++r)
    copy(it->begin(), it->end(), static_cast<npy_double*>(PyArray_GETPTR2(reinterpret_cast<PyArrayObject*>(ret.get()), r, 0)));
  return C(ret);
}

Eval::Value PyEval::create_string(const string& v) const {
  return C(CALLPY(PyUnicode_FromString, v));
}

Eval::Value PyEval::createFunctionDep(const vector<Value>& v) const {
  PyO arg(CALLPY(PyTuple_New, 1));
  PyO item(CALLPY(PyList_New, v.size()));
  for(size_t i=0; i<v.size(); ++i)
    CALLPY(PyList_SetItem, item, i, C(v[i]).incRef());
  CALLPY(PyTuple_SetItem, arg, 0, item.incRef());
  return C(CALLPY(PyObject_CallObject, pyInit->matrix(), arg));
}

Eval::Value PyEval::createFunctionDep(const vector<vector<Value> >& v) const {
  PyO arg(CALLPY(PyTuple_New, 1));
  PyO rows(CALLPY(PyList_New, v.size()));
  for(size_t r=0; r<v.size(); ++r) {
    PyO row(CALLPY(PyList_New, v[r].size()));
    for(size_t c=0; c<v[r].size(); ++c)
      CALLPY(PyList_SetItem, row, c, C(v[r][c]).incRef());
    CALLPY(PyList_SetItem, rows, r, row.incRef());
  }
  CALLPY(PyTuple_SetItem, arg, 0, rows.incRef());
  return C(CALLPY(PyObject_CallObject, pyInit->matrix(), arg));
}

Eval::Value PyEval::createFunction(const vector<Value> &indeps, const Value &dep) const {
  auto t=CALLPY(PyTuple_New, indeps.size()+1);
  for(size_t i=0; i<indeps.size(); ++i)
    CALLPY(PyTuple_SetItem, t, i, C(indeps[i]).incRef());
  CALLPY(PyTuple_SetItem, t, indeps.size(), C(dep).incRef());
  return C(t);
}

string PyEval::serializeFunction(const Value &x) const {
  auto serializeFunctionPy=[](PyO& o) {
    PyO arg(CALLPY(PyTuple_New, 1));
    CALLPY(PyTuple_SetItem, arg, 0, o.incRef());
    PyO ser=CALLPY(PyObject_CallObject, pyInit->serializeFunction(), arg);
    return string(CALLPY(PyUnicode_AsUTF8, ser));
  };

  auto nrIndeps=CALLPY(PyTuple_Size, C(x))-1;
  string ret("f(");
  for(size_t i=0; i<nrIndeps; ++i) {
    auto indep=CALLPYB(PyTuple_GetItem, C(x), i);
    ret+=(i==0?"":",")+serializeFunctionPy(indep);
  }
  ret+=")=";
  auto dep=CALLPYB(PyTuple_GetItem, C(x), nrIndeps);
  ret+=serializeFunctionPy(dep);
  return ret;
}

void PyEval::convertIndex(Value &v, bool evalTo1Base) {
  int add = evalTo1Base ? 1 : -1;

  if(valueIsOfType(v, ScalarType))
    v=create_double(cast_double(v)+add);
  else if(valueIsOfType(v, VectorType)) {
    PyArrayObject *a=reinterpret_cast<PyArrayObject*>(C(v).get());
    int type=PyArray_TYPE(a);
    if(type!=NPY_SHORT    && type!=NPY_USHORT    &&
       type!=NPY_INT      && type!=NPY_UINT      &&
       type!=NPY_LONG     && type!=NPY_ULONG     &&
       type!=NPY_LONGLONG && type!=NPY_ULONGLONG)
      throw runtime_error("Value is not of type integer.");
    npy_intp *dims=PyArray_SHAPE(a);
    switch(type) {
      case NPY_SHORT:     for(size_t i=0; i<dims[0]; ++i) *static_cast<npy_short*>    (PyArray_GETPTR1(a, i)) += add; break;
      case NPY_USHORT:    for(size_t i=0; i<dims[0]; ++i) *static_cast<npy_ushort*>   (PyArray_GETPTR1(a, i)) += add; break;
      case NPY_INT:       for(size_t i=0; i<dims[0]; ++i) *static_cast<npy_int*>      (PyArray_GETPTR1(a, i)) += add; break;
      case NPY_UINT:      for(size_t i=0; i<dims[0]; ++i) *static_cast<npy_uint*>     (PyArray_GETPTR1(a, i)) += add; break;
      case NPY_LONG:      for(size_t i=0; i<dims[0]; ++i) *static_cast<npy_long*>     (PyArray_GETPTR1(a, i)) += add; break;
      case NPY_ULONG:     for(size_t i=0; i<dims[0]; ++i) *static_cast<npy_ulong*>    (PyArray_GETPTR1(a, i)) += add; break;
      case NPY_LONGLONG:  for(size_t i=0; i<dims[0]; ++i) *static_cast<npy_longlong*> (PyArray_GETPTR1(a, i)) += add; break;
      case NPY_ULONGLONG: for(size_t i=0; i<dims[0]; ++i) *static_cast<npy_ulonglong*>(PyArray_GETPTR1(a, i)) += add; break;
    }
  }
  else
    throw runtime_error("Only scalars and vectors can be handled as indices.");
}

}

namespace {

bool checkNumPyDoubleType(int type) {
  if(type!=NPY_BOOL &&
     type!=NPY_BYTE &&
     type!=NPY_UBYTE &&
     type!=NPY_SHORT &&
     type!=NPY_USHORT &&
     type!=NPY_INT &&
     type!=NPY_UINT &&
     type!=NPY_LONG &&
     type!=NPY_ULONG &&
     type!=NPY_LONGLONG &&
     type!=NPY_ULONGLONG &&
     type!=NPY_FLOAT &&
     type!=NPY_DOUBLE &&
     type!=NPY_LONGDOUBLE)
    return false;
  return true;
}

double arrayGetDouble(PyArrayObject *a, int type, int r, int c) {
  switch(type) {
    case NPY_BOOL:       return *static_cast<npy_bool*>      (c==-1 ? PyArray_GETPTR1(a, r) : PyArray_GETPTR2(a, r, c));
    case NPY_BYTE:       return *static_cast<npy_byte*>      (c==-1 ? PyArray_GETPTR1(a, r) : PyArray_GETPTR2(a, r, c));
    case NPY_UBYTE:      return *static_cast<npy_ubyte*>     (c==-1 ? PyArray_GETPTR1(a, r) : PyArray_GETPTR2(a, r, c));
    case NPY_SHORT:      return *static_cast<npy_short*>     (c==-1 ? PyArray_GETPTR1(a, r) : PyArray_GETPTR2(a, r, c));
    case NPY_USHORT:     return *static_cast<npy_ushort*>    (c==-1 ? PyArray_GETPTR1(a, r) : PyArray_GETPTR2(a, r, c));
    case NPY_INT:        return *static_cast<npy_int*>       (c==-1 ? PyArray_GETPTR1(a, r) : PyArray_GETPTR2(a, r, c));
    case NPY_UINT:       return *static_cast<npy_uint*>      (c==-1 ? PyArray_GETPTR1(a, r) : PyArray_GETPTR2(a, r, c));
    case NPY_LONG:       return *static_cast<npy_long*>      (c==-1 ? PyArray_GETPTR1(a, r) : PyArray_GETPTR2(a, r, c));
    case NPY_ULONG:      return *static_cast<npy_ulong*>     (c==-1 ? PyArray_GETPTR1(a, r) : PyArray_GETPTR2(a, r, c));
    case NPY_LONGLONG:   return *static_cast<npy_longlong*>  (c==-1 ? PyArray_GETPTR1(a, r) : PyArray_GETPTR2(a, r, c));
    case NPY_ULONGLONG:  return *static_cast<npy_ulonglong*> (c==-1 ? PyArray_GETPTR1(a, r) : PyArray_GETPTR2(a, r, c));
    case NPY_FLOAT:      return *static_cast<npy_float*>     (c==-1 ? PyArray_GETPTR1(a, r) : PyArray_GETPTR2(a, r, c));
    case NPY_DOUBLE:     return *static_cast<npy_double*>    (c==-1 ? PyArray_GETPTR1(a, r) : PyArray_GETPTR2(a, r, c));
    case NPY_LONGDOUBLE: return *static_cast<npy_longdouble*>(c==-1 ? PyArray_GETPTR1(a, r) : PyArray_GETPTR2(a, r, c));
  }
  throw runtime_error("Value is not of type double (wrong element type).");
}

double arrayScalarGetDouble(PyObject *o, bool *error) {
  double ret;
  PyArray_Descr *descr=PyArray_DescrFromScalar(o);
  if(     descr->typeobj==&PyBoolArrType_Type)        { npy_bool        v; PyArray_ScalarAsCtype(o, &v); ret=v; }
  else if(descr->typeobj==&PyByteArrType_Type)        { npy_byte        v; PyArray_ScalarAsCtype(o, &v); ret=v; }
  else if(descr->typeobj==&PyUByteArrType_Type)       { npy_ubyte       v; PyArray_ScalarAsCtype(o, &v); ret=v; }
  else if(descr->typeobj==&PyShortArrType_Type)       { npy_short       v; PyArray_ScalarAsCtype(o, &v); ret=v; }
  else if(descr->typeobj==&PyUShortArrType_Type)      { npy_ushort      v; PyArray_ScalarAsCtype(o, &v); ret=v; }
  else if(descr->typeobj==&PyIntArrType_Type)         { npy_int         v; PyArray_ScalarAsCtype(o, &v); ret=v; }
  else if(descr->typeobj==&PyUIntArrType_Type)        { npy_uint        v; PyArray_ScalarAsCtype(o, &v); ret=v; }
  else if(descr->typeobj==&PyLongArrType_Type)        { npy_long        v; PyArray_ScalarAsCtype(o, &v); ret=v; }
  else if(descr->typeobj==&PyULongArrType_Type)       { npy_ulong       v; PyArray_ScalarAsCtype(o, &v); ret=v; }
  else if(descr->typeobj==&PyLongLongArrType_Type)    { npy_longlong    v; PyArray_ScalarAsCtype(o, &v); ret=v; }
  else if(descr->typeobj==&PyULongLongArrType_Type)   { npy_ulonglong   v; PyArray_ScalarAsCtype(o, &v); ret=v; }
  else if(descr->typeobj==&PyFloatArrType_Type)       { npy_float       v; PyArray_ScalarAsCtype(o, &v); ret=v; }
  else if(descr->typeobj==&PyDoubleArrType_Type)      { npy_double      v; PyArray_ScalarAsCtype(o, &v); ret=v; }
  else if(descr->typeobj==&PyLongDoubleArrType_Type)  { npy_longdouble  v; PyArray_ScalarAsCtype(o, &v); ret=v; }
  else {
    Py_DECREF(descr);
    if(!error)
      throw runtime_error("Internal error: unknown type.");
    *error=true;
    return 0;
  }
  Py_DECREF(descr);
  if(error) *error=false;
  return ret;
}

bool is_scalar_double(const MBXMLUtils::Eval::Value &value) {
  PyO v(C(value));
  if(CALLPY(PyFloat_Check, v) || CALLPY(PyLong_Check, v) || CALLPY(PyBool_Check, v))
    return true;
  if(PyArray_CheckScalar(v.get())) {
    bool error;
    arrayScalarGetDouble(v.get(), &error);
    return !error;
  }
  return false;
}

bool is_vector_double(const MBXMLUtils::Eval::Value &value, PyArrayObject** a, int *type) {
  PyO v(C(value));
  if(PyArray_Check(v.get())) {
    auto *a_=reinterpret_cast<PyArrayObject*>(v.get());
    if(a) *a=a_;
    if(PyArray_NDIM(a_)!=1)
      return false;
    int type_=PyArray_TYPE(a_);
    if(type) *type=type_;
    if(!checkNumPyDoubleType(type_))
      return false;
    return true;
  }
  return false;
}

bool is_vector_vector_double(const MBXMLUtils::Eval::Value &value, PyArrayObject **a, int *type) {
  PyO v(C(value));
  if(PyArray_Check(v.get())) {
    auto *a_=reinterpret_cast<PyArrayObject*>(v.get());
    if(a) *a=a_;
    if(PyArray_NDIM(a_)!=2)
      return false;
    int type_=PyArray_TYPE(a_);
    if(type) *type=type_;
    if(!checkNumPyDoubleType(type_))
      return false;
    return true;
  }
  return false;
}

}

// called from mbxmlutils.registerPath and adds path to the dependencies of this evaluator
extern "C" int mbxmlutilsPyEvalRegisterPath(const char *path) {
  mbxmlutilsStaticDependencies.emplace_back(path);
  return 0;
}

// called from mbxmlutils.getOriginalFilename and returns the filename of the currently evaluated element
extern "C" const char* mbxmlutilsPyEvalGetOriginalFilename() {
  static string originalFilenameStr;
  originalFilenameStr = originalFilename.string();
  return originalFilenameStr.c_str();
}
