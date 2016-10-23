#include <config.h>

// python includes
#include <Python.h> // due to some bugs in python 3.2 we need to include this first
#define PY_ARRAY_UNIQUE_SYMBOL mbxmlutils_pyeval_ARRAY_API
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

// normal includes
#include "pyeval.h"

// other includes
#include <boost/lexical_cast.hpp> // to convert a double to int and throw if its not an int
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/join.hpp>
#include <mbxmlutilshelper/getinstallpath.h>
#include "mbxmlutils/eval_static.h"
#include "pyeval-config.h"

using namespace std;
using namespace boost::filesystem;
using namespace xercesc;
using namespace PythonCpp;

namespace {

inline PyO C(const MBXMLUtils::Eval::Value &value) {
  return *static_pointer_cast<PyO>(boost::get<shared_ptr<void> >(value));
}

inline MBXMLUtils::Eval::Value C(const PyO &value) {
  return make_shared<PyO>(value);
}

vector<double> cast_vector_double(const MBXMLUtils::Eval::Value &value, bool checkOnly);
vector<vector<double> > cast_vector_vector_double(const MBXMLUtils::Eval::Value &value, bool checkOnly);
string cast_string(const MBXMLUtils::Eval::Value &value, bool checkOnly);
double arrayScalarGetDouble(PyObject *o);

}

namespace MBXMLUtils {

XMLUTILS_EVAL_REGISTER(PyEval)

// Helper class to init/deinit Python on library load/unload (unload=program end)
class PyInit {
  public:
    PyInit();
    ~PyInit();
    PyO mbxmlutils;
    PyO casadiValue;
    map<string, PyO> functionValue;
    PyO asarray;
};

PyInit::PyInit() {
  try {
    initializePython((getInstallPath()/"bin"/"mbxmlutilspp").string());
    CALLPY(_import_array);

    PyO path(CALLPYB(PySys_GetObject, const_cast<char*>("path")));
    PyO mbxmlutilspath(CALLPY(PyUnicode_FromString, (getInstallPath()/"share"/"mbxmlutils"/"python").string()));
    CALLPY(PyList_Append, path, mbxmlutilspath);
    PyO casadipath(CALLPY(PyUnicode_FromString, CASADI_PREFIX_DIR "/python2.7/site-packages"));
    CALLPY(PyList_Append, path, casadipath);

    mbxmlutils=CALLPY(PyImport_ImportModule, "mbxmlutils");

    casadiValue=CALLPY(PyImport_ImportModule, "casadi");

    PyO numpy(CALLPY(PyImport_ImportModule, "numpy"));
    asarray=CALLPY(PyObject_GetAttrString, numpy, "asarray");
  }
  // print error to cerr and rethrow. (The exception may not be cached since this is called in pre-main)
  catch(const std::exception& ex) {
    cerr<<"Exception during Python initialization:"<<endl<<ex.what()<<endl;
    throw;
  }
  catch(...) {
    cerr<<"Unknown exception during Python initialization."<<endl;
    throw;
  }
}

PyInit::~PyInit() {
  try {
    // clear all Python object before deinit
    asarray.reset();
    functionValue.clear();
    casadiValue.reset();
    mbxmlutils.reset();
  }
  // print error to cerr and rethrow. (The exception may not be cached since this is called in pre-main)
  catch(const std::exception& ex) {
    cerr<<"Exception during Python deinitialization:"<<endl<<ex.what()<<endl;
    throw;
  }
  catch(...) {
    cerr<<"Unknown exception during Python deinitialization."<<endl;
    throw;
  }
}

PyInit pyInit; // init Python on library load and deinit on library unload = program end

PyEval::PyEval(vector<path> *dependencies_) : Eval(dependencies_) {
  casadiValue=C(pyInit.casadiValue);
  currentImport=make_shared<PyO>(CALLPY(PyDict_New));
}

PyEval::~PyEval() {
}

void PyEval::addImport(const string &code, const DOMElement *e) {
  // restore current dir on exit and change current dir
  PreserveCurrentDir preserveDir;
  if(e) {
    path chdir=E(e)->getOriginalFilename().parent_path();
    if(!chdir.empty())
      current_path(chdir);
  }

  // python globals (fill with builtins)
  PyO globals(CALLPY(PyDict_New));
  CALLPY(PyDict_SetItemString, globals, "__builtins__", CALLPYB(PyEval_GetBuiltins));
  // python globals (fill with current parameters)
  for(unordered_map<string, Value>::const_iterator i=currentParam.begin(); i!=currentParam.end(); i++)
    CALLPY(PyDict_SetItemString, globals, i->first, C(i->second));

  // evaluate as statement
  PyO locals(CALLPY(PyDict_New));
  CALLPY(PyRun_String, code, Py_file_input, globals, locals);

  // get all locals and add to currentImport
  CALLPY(PyDict_Merge, *static_pointer_cast<PyO>(currentImport), locals, true);
}

bool PyEval::valueIsOfType(const Value &value, ValueType type) const {
  if(type==FunctionType && boost::get<Function>(&value))
    return true;
  PyO v(C(value));
  switch(type) {
    case ScalarType: return CALLPY(PyFloat_Check, v) || CALLPY(PyLong_Check, v) || CALLPY(PyBool_Check, v);
    case VectorType: try { ::cast_vector_double(value, true); return true; } catch(...) { return false; }
    case MatrixType: try { ::cast_vector_vector_double(value, true); return true; } catch(...) { return false; }
    case StringType: try { ::cast_string(value, true); return true; } catch(...) { return false; }
    case FunctionType: return false;
  }
  throw DOMEvalException("Internal error: Unknwon ValueType.");
}

map<path, pair<path, bool> >& PyEval::requiredFiles() const {
  static map<path, pair<path, bool> > files;
  return files;
}

Eval::Value PyEval::createSwigByTypeName(const string &typeName) const {
  return C(CALLPY(PyObject_CallObject, CALLPYB(PyDict_GetItemString, CALLPYB(PyModule_GetDict, C(casadiValue)), typeName), PyO()));
}

Eval::Value PyEval::callFunction(const string &name, const vector<Value>& args) const {
  pair<map<string, PyO>::iterator, bool> f=pyInit.functionValue.insert(make_pair(name, PyO()));
  if(f.second)
    f.first->second=CALLPYB(PyDict_GetItemString, CALLPYB(PyModule_GetDict, pyInit.mbxmlutils), name);
  PyO pyargs(CALLPY(PyTuple_New, args.size()));
  int idx=0;
  for(vector<Value>::const_iterator it=args.begin(); it!=args.end(); ++it, ++idx) {
    PyO v(C(*it));
    CALLPY(PyTuple_SetItem, pyargs, idx, v.incRef()); // PyTuple_SetItem steals a reference of v
  }
  return C(CALLPY(PyObject_CallObject, f.first->second, pyargs));
}

Eval::Value PyEval::fullStringToValue(const string &str, const DOMElement *e) const {
  string strtrim=str;
  boost::trim(strtrim);

  // check some common string to avoid time consiming evaluation
  // check true and false
  if(strtrim=="True") return C(CALLPY(PyBool_FromLong, 1));
  if(strtrim=="False") return C(CALLPY(PyBool_FromLong, 0));
  // check for integer and floating point values
  try { return C(CALLPY(PyLong_FromLong, boost::lexical_cast<int>(boost::algorithm::trim_copy(strtrim)))); }
  catch(const boost::bad_lexical_cast &) {}
  try { return C(CALLPY(PyFloat_FromDouble, boost::lexical_cast<double>(boost::algorithm::trim_copy(strtrim)))); }
  catch(const boost::bad_lexical_cast &) {}
  // no common string detected -> evaluate using python now

  // restore current dir on exit and change current dir
  PreserveCurrentDir preserveDir;
  if(e) {
    path chdir=E(e)->getOriginalFilename().parent_path();
    if(!chdir.empty())
      current_path(chdir);
  }

  // python globals (fill with builtins)
  PyO globals(CALLPY(PyDict_New));
  CALLPY(PyDict_SetItemString, globals, "__builtins__", CALLPYB(PyEval_GetBuiltins));
  // python globals (fill with imports)
  CALLPY(PyDict_Merge, globals, *static_pointer_cast<PyO>(currentImport), true);
  // python globals (fill with current parameters)
  for(unordered_map<string, Value>::const_iterator i=currentParam.begin(); i!=currentParam.end(); i++)
    CALLPY(PyDict_SetItemString, globals, i->first, C(i->second));

  PyO ret;
  PyCompilerFlags flags;
  flags.cf_flags=CO_FUTURE_DIVISION; // we evaluate the code in python 3 mode (future python 2 mode)
  try {
    // evaluate as expression (using the trimmed str) and save result in ret
    PyO locals(CALLPY(PyDict_New));
    mbxmlutilsStaticDependencies.clear();
    ret=CALLPY(PyRun_StringFlags, strtrim, Py_eval_input, globals, locals, &flags);
    addStaticDependencies(e);
  }
  catch(const std::exception&) { // on failure ...
    PyO locals(CALLPY(PyDict_New));
    try {
      // ... evaluate as statement

      // fix python indentation
      vector<string> lines;
      boost::split(lines, str, boost::is_any_of("\n")); // split to a vector of lines
      size_t indent=string::npos;
      size_t lineNr=0;
      for(vector<string>::iterator it=lines.begin(); it!=lines.end(); ++it, ++lineNr) {
        size_t pos=it->find_first_not_of(' '); // get first none space character
        if(pos==string::npos) continue; // not found -> pure empty line -> do not modify
        if(pos!=string::npos && (*it)[pos]=='#') continue; // found and first char is '#' -> pure comment line -> do not modify
        // now we have a line with a python statement
        if(indent==string::npos) indent=pos; // at the first python statement line use the current indent as indent for all others
        if(it->substr(0, indent)!=string(indent, ' ')) // check if line starts with at least indent spaces ...
          // ... if not its an indentation error
          throw DOMEvalException("Unexpected indentation at line "+to_string(lineNr)+": "+str, e);
        *it=it->substr(indent); // remove the first indent spaces from the line
      }
      strtrim=boost::join(lines, "\n"); // join the lines to a single string

      // evaluate as statement
      mbxmlutilsStaticDependencies.clear();
      CALLPY(PyRun_StringFlags, strtrim, Py_file_input, globals, locals, &flags);
      addStaticDependencies(e);
    }
    catch(const std::exception& ex) { // on failure -> report error
      throw DOMEvalException(string(ex.what())+"Unable to evaluate expression:\n"+str, e);
    }
    try {
      // get 'ret' variable from statement
      ret=CALLPYB(PyDict_GetItemString, locals, "ret");
    }
    catch(const std::exception&) {
      // 'ret' variable not found or invalid expression
      throw DOMEvalException("Invalid expression or statement does not define the 'ret' variable in expression:\nX"+str, e);
    }
  }
  // convert a list or list of lists to a numpy array
  if(CALLPY(PyList_Check, ret)) {
    PyO args(CALLPY(PyTuple_New, 1));
    CALLPY(PyTuple_SetItem, args, 0, ret.incRef()); // PyTuple_SetItem steals a reference of ret
    return C(CALLPY(PyObject_CallObject, pyInit.asarray, args));
  }
  // return result
  return C(ret);
}

string PyEval::getSwigType(const Value &value) const {
  return C(value)->ob_type->tp_name;
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
  return ::cast_vector_double(value, false);
}

vector<vector<double> >PyEval::cast_vector_vector_double(const Value &value) const {
  return ::cast_vector_vector_double(value, false);
}

string PyEval::cast_string(const Value &value) const {
  return ::cast_string(value, false);
}

Eval::Value PyEval::create_double(const double& v) const {
  try { return C(CALLPY(PyLong_FromLong, boost::lexical_cast<int>(v))); } catch(const boost::bad_lexical_cast &) {}
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
  for(vector<vector<double> >::const_iterator it=v.begin(); it!=v.end(); ++it, ++r)
    copy(it->begin(), it->end(), static_cast<npy_double*>(PyArray_GETPTR2(reinterpret_cast<PyArrayObject*>(ret.get()), r, 0)));
  return C(ret);
}

Eval::Value PyEval::create_string(const string& v) const {
  return C(CALLPY(PyUnicode_FromString, v));
}

}

namespace {

void checkNumPyDoubleType(int type) {
  if(type!=NPY_SHORT    && type!=NPY_USHORT    &&
     type!=NPY_INT      && type!=NPY_UINT      &&
     type!=NPY_LONG     && type!=NPY_ULONG     &&
     type!=NPY_LONGLONG && type!=NPY_ULONGLONG &&
     type!=NPY_FLOAT    && type!=NPY_DOUBLE    && type!=NPY_LONGDOUBLE)
    throw runtime_error("Value is not of type double.");
}

double arrayGetDouble(PyArrayObject *a, int type, int r, int c=-1) {
  switch(type) {
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

double arrayScalarGetDouble(PyObject *o) {
  double ret;
  PyArray_Descr *descr=PyArray_DescrFromScalar(o);
  if(     descr->typeobj==&PyBoolArrType_Type)        { npy_bool        v; PyArray_ScalarAsCtype(o, &v); ret=v; }
  else if(descr->typeobj==&PyByteArrType_Type)        { npy_byte        v; PyArray_ScalarAsCtype(o, &v); ret=v; }
  else if(descr->typeobj==&PyShortArrType_Type)       { npy_short       v; PyArray_ScalarAsCtype(o, &v); ret=v; }
  else if(descr->typeobj==&PyIntArrType_Type)         { npy_int         v; PyArray_ScalarAsCtype(o, &v); ret=v; }
  else if(descr->typeobj==&PyLongArrType_Type)        { npy_long        v; PyArray_ScalarAsCtype(o, &v); ret=v; }
  else if(descr->typeobj==&PyLongLongArrType_Type)    { npy_longlong    v; PyArray_ScalarAsCtype(o, &v); ret=v; }
  else if(descr->typeobj==&PyUByteArrType_Type)       { npy_ubyte       v; PyArray_ScalarAsCtype(o, &v); ret=v; }
  else if(descr->typeobj==&PyUShortArrType_Type)      { npy_ushort      v; PyArray_ScalarAsCtype(o, &v); ret=v; }
  else if(descr->typeobj==&PyUIntArrType_Type)        { npy_uint        v; PyArray_ScalarAsCtype(o, &v); ret=v; }
  else if(descr->typeobj==&PyULongArrType_Type)       { npy_ulong       v; PyArray_ScalarAsCtype(o, &v); ret=v; }
  else if(descr->typeobj==&PyULongLongArrType_Type)   { npy_ulonglong   v; PyArray_ScalarAsCtype(o, &v); ret=v; }
  else if(descr->typeobj==&PyFloatArrType_Type)       { npy_float       v; PyArray_ScalarAsCtype(o, &v); ret=v; }
  else if(descr->typeobj==&PyDoubleArrType_Type)      { npy_double      v; PyArray_ScalarAsCtype(o, &v); ret=v; }
  else if(descr->typeobj==&PyLongDoubleArrType_Type)  { npy_longdouble  v; PyArray_ScalarAsCtype(o, &v); ret=v; }
  else { Py_DECREF(descr); throw runtime_error("Internal error: unknown type."); }
  Py_DECREF(descr);
  return ret;
}

vector<double> cast_vector_double(const MBXMLUtils::Eval::Value &value, bool checkOnly) {
  PyO v(C(value));
  if(PyArray_Check(v.get())) {
    PyArrayObject *a=reinterpret_cast<PyArrayObject*>(v.get());
    if(PyArray_NDIM(a)!=1)
      throw runtime_error("Value is not of type vector (wrong dimension).");
    int type=PyArray_TYPE(a);
    checkNumPyDoubleType(type);
    if(checkOnly)
      return vector<double>();
    npy_intp *dims=PyArray_SHAPE(a);
    vector<double> ret(dims[0]);
    for(size_t i=0; i<dims[0]; ++i)
      ret[i]=arrayGetDouble(a, type, i);
    return ret;
  }
  throw runtime_error("Value is not of type vector (wrong type).");
}

vector<vector<double> > cast_vector_vector_double(const MBXMLUtils::Eval::Value &value, bool checkOnly) {
  PyO v(C(value));
  if(PyArray_Check(v.get())) {
    PyArrayObject *a=reinterpret_cast<PyArrayObject*>(v.get());
    if(PyArray_NDIM(a)!=2)
      throw runtime_error("Value is not of type matrix (wrong dimension).");
    int type=PyArray_TYPE(a);
    checkNumPyDoubleType(type);
    if(checkOnly)
      return vector<vector<double> >();
    npy_intp *dims=PyArray_SHAPE(a);
    vector<vector<double> > ret(dims[0], vector<double>(dims[1]));
    for(size_t r=0; r<dims[0]; ++r)
      for(size_t c=0; c<dims[1]; ++c)
        ret[r][c]=arrayGetDouble(a, type, r, c);
    return ret;
  }
  throw runtime_error("Value is not of type matrix (wrong type).");
}

string cast_string(const MBXMLUtils::Eval::Value &value, bool checkOnly) {
  return CALLPY(PyUnicode_AsUTF8, C(value));
}


}

// called from mbxmlutils.registerPath and adds path to the dependencies of this evaluator
extern "C" int mbxmlutilsPyEvalRegisterPath(const char *path) {
  mbxmlutilsStaticDependencies.push_back(path);
  return 0;
}

namespace MBXMLUtils {

// We include swigpyrun.h at the end here to avoid the usage of functions macros
// defined in this file which we do not want to use.

#include "swigpyrun.h"

void* PyEval::getSwigThis(const Value &value) const {
  return SWIG_Python_GetSwigThis(C(value).get())->ptr;
}

}
