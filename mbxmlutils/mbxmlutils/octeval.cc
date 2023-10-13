// octave used M_PI which is not longer defined in newer compilers
#define _USE_MATH_DEFINES
#include <cmath>

// includes are somehow tricky with octave, see also Makefile.am

// include config.h
// we cannot add -I.. to the compiler options, see Makefile.am, hence we add -I.. and
// use this to include config.h
#include <../mbxmlutils/../config.h>

// normal includes
#include <functional>
#include <mbxmlutilshelper/dom.h>
#include <fmatvec/fmatvec.h>
#include <fmatvec/ast.h>
#include <xercesc/dom/DOMAttr.hpp>
#include "mbxmlutils/octeval.h"
#include "mbxmlutils/eval_static.h"
#include <boost/algorithm/string/trim.hpp>

#include <octave-config.h>
#undef OCTAVE_USE_DEPRECATED_FUNCTIONS
#include <octave/ovl.h>
#include <octave/interpreter.h>
#include <octave/octave.h>
#include <octave/version.h>
#include <octave/symtab.h>
#include <octave/parse.h>
#include <octave/defaults.h>

using namespace xercesc;
namespace bfs=boost::filesystem;

namespace {
  //TODO not working on Windows
  //TODO // NOTE: we can skip the use of utf8Facet (see below) and set the facet globally (for bfs::path and others) using:
  //TODO // std::locale::global(locale::generator().generate("UTF8"));
  //TODO // filesystem::path::imbue(std::locale());
  //TODO const bfs::path::codecvt_type *utf8Facet(&use_facet<bfs::path::codecvt_type>(locale::generator().generate("UTF8")));
  #define CODECVT

  // some platform dependent values
#ifdef _WIN32
  const bfs::path LIBDIR="bin";
#else
  const bfs::path LIBDIR="lib";
#endif

  const bool deactivateBlock=getenv("MBXMLUTILS_DEACTIVATE_BLOCK")!=nullptr;

  // A class to block/unblock stderr or stdout. Block in called in the ctor, unblock in the dtor
  template<int T>
  class Block {
    public:
      Block(std::ostream &str_, std::streambuf *buf=nullptr) : str(str_) {
        if(deactivateBlock) return;
        if(disableCount==0)
          orgcxxx=str.rdbuf(buf);
        disableCount++;
      }
      ~Block() {
        if(deactivateBlock) return;
        disableCount--;
        if(disableCount==0)
          str.rdbuf(orgcxxx);
      }
    private:
      std::ostream &str;
      static std::streambuf *orgcxxx;
      static int disableCount;
  };
  template<int T> std::streambuf *Block<T>::orgcxxx;
  template<int T> int Block<T>::disableCount=0;
  #define BLOCK_STDOUT Block<1> MBXMLUTILS_EVAL_CONCAT(mbxmlutils_blockstdout_, __LINE__)(std::cout)
  #define BLOCK_STDERR Block<2> MBXMLUTILS_EVAL_CONCAT(mbxmlutils_blockstderr_, __LINE__)(std::cerr)
  #define REDIR_STDOUT(buf) Block<1> MBXMLUTILS_EVAL_CONCAT(mbxmlutils_redirstdout_, __LINE__)(std::cout, buf)
  #define REDIR_STDERR(buf) Block<2> MBXMLUTILS_EVAL_CONCAT(mbxmlutils_redirstderr_, __LINE__)(std::cerr, buf)
}

namespace MBXMLUtils {

MBXMLUTILS_EVAL_REGISTER(OctEval)

// Helper class to init/deinit octave on library load/unload (unload=program end)
class OctInit {
  public:
    OctInit();
    ~OctInit();
    std::string initialPath;
    octave::interpreter *interpreter { nullptr };
};

OctInit::OctInit() {
  try {
    // set the OCTAVE_HOME envvar and octave_prefix variable before initializing octave
    bfs::path octave_prefix;
    if(getenv("OCTAVE_HOME")) // OCTAVE_HOME set manually -> use this for octave_prefix
      octave_prefix=getenv("OCTAVE_HOME");
    else if(getenv("OCTAVE_HOME")==nullptr && bfs::exists(MBXMLUtils::Eval::installPath/"share"/"octave")) {
      // OCTAVE_HOME not set but octave is available in installation path of MBXMLUtils -> use installation path
      octave_prefix=MBXMLUtils::Eval::installPath;
      // the string for putenv must have program life time
      static std::string OCTAVE_HOME="OCTAVE_HOME="+MBXMLUtils::Eval::installPath.string(CODECVT);
      putenv((char*)OCTAVE_HOME.c_str());
    }
    // init interpreter
    interpreter=new octave::interpreter();
    if(octave_prefix.empty())
      octave_prefix=octave::config::octave_home();

    if(interpreter->execute()!=0)
      throw std::runtime_error("Cannot execute octave interpreter.");
  
    // set some global octave config
    octave_value_list warnArg;
    warnArg.append("error");
    warnArg.append("Octave:divide-by-zero");
    octave::feval("warning", warnArg);

    // ... and add .../[bin|lib] to octave search path (their we push all oct files)
    std::string dir=(MBXMLUtils::Eval::installPath/LIBDIR).string(CODECVT);
    octave::feval("addpath", octave_value_list(octave_value(dir)));

    // save initial octave search path
    initialPath=octave::feval("path", octave_value_list(), 1)(0).string_value();

    // deregister __finish__, see OctEval::~OctEval
    octave_value_list atexitArg;
    atexitArg.append("__finish__");
    atexitArg.append(0);
    octave::feval("atexit", atexitArg);
  }
  // print error and rethrow. (The exception may not be catched since this is called in pre-main)
  catch(octave::execution_exception &ex)
  {
    // octave::execution_exception is not derived (in older releases) from std::exception -> convert it to an std::exception.
    fmatvec::Atom::msgStatic(fmatvec::Atom::Error)<<"Exception during octave initialization:"<<std::endl<<ex.
#if OCTAVE_MAJOR_VERSION >= 6
      message()
#else
      info()
#endif
      <<std::endl;
    throw;
  }
  catch(octave::exit_exception &ex)
  {
    // octave::exit_exception is not derived (in older releases) from std::exception -> convert it to an std::exception.
    fmatvec::Atom::msgStatic(fmatvec::Atom::Error)<<"Exception during octave initialization:"<<std::endl<<"Exit exception"<<std::endl;
    throw;
  }
  catch(octave::interrupt_exception &ex)
  {
    // octave::interrupt_exception is not derived (in older releases) from std::exception -> convert it to an std::exception.
    fmatvec::Atom::msgStatic(fmatvec::Atom::Error)<<"Exception during octave initialization:"<<std::endl<<"Interrupt exception"<<std::endl;
    throw;
  }
  catch(const std::exception& ex) {
    fmatvec::Atom::msgStatic(fmatvec::Atom::Error)<<"Exception during octave initialization:"<<std::endl<<ex.what()<<std::endl;
    throw;
  }
  catch(...) {
    fmatvec::Atom::msgStatic(fmatvec::Atom::Error)<<"Unknown exception during octave initialization."<<std::endl;
    throw;
  }
}

OctInit::~OctInit() {
  try {
#if OCTAVE_MAJOR_VERSION >= 6
    // deinit with octave 6.2 causes several invalid read memory access -> disabled deinit for octave 6.2
    // We need to live with a lot of octave memory leak at exit of the program. But this is not a real problem.
    //interpreter->shutdown();
    //delete interpreter;
#else
    delete interpreter;
#endif

    // __finish__.m which is run at exit is deregistered during octave initialzation via atext.
    // This is required to avoid running octave code after octave was alread removed.
    // (atexit is run on library unload at program exit)
  }
  // print error and rethrow. (The exception may not be catched since this is called in pre-main)
  catch(octave::execution_exception &ex)
  {
    // octave::execution_exception is not derived (in older release) from std::exception -> convert it to an std::exception.
    fmatvec::Atom::msgStatic(fmatvec::Atom::Error)<<"Exception during octave deinitialization:"<<std::endl<<ex.
#if OCTAVE_MAJOR_VERSION >= 6
      message()
#else
      info()
#endif
      <<std::endl
      <<"Continuing but undefined behaviour may occur."<<std::endl;
  }
  catch(octave::exit_exception &ex)
  {
    // octave::exit_exception is not derived (in older release) from std::exception -> convert it to an std::exception.
    fmatvec::Atom::msgStatic(fmatvec::Atom::Error)<<"Exception during octave deinitialization:"<<std::endl<<"Exit exception"<<std::endl
      <<"Continuing but undefined behaviour may occur."<<std::endl;
  }
  catch(octave::interrupt_exception &ex)
  {
    // octave::interrupt_exception is not derived (in older release) from std::exception -> convert it to an std::exception.
    fmatvec::Atom::msgStatic(fmatvec::Atom::Error)<<"Exception during octave deinitialization:"<<std::endl<<"Interrupt exception"<<std::endl
      <<"Continuing but undefined behaviour may occur."<<std::endl;
  }
  catch(const std::exception& ex) {
    fmatvec::Atom::msgStatic(fmatvec::Atom::Error)<<"Exception during octave deinitialization:"<<std::endl<<ex.what()<<std::endl
      <<"Continuing but undefined behaviour may occur."<<std::endl;
  }
  catch(...) {
    fmatvec::Atom::msgStatic(fmatvec::Atom::Error)<<"Unknown exception during octave deinitialization."<<std::endl
      <<"Continuing but undefined behaviour may occur."<<std::endl;
  }
}

OctInit octInit; // init octave on library load and deinit on library unload = program end

inline std::shared_ptr<octave_value> C(const Eval::Value &value) {
  return std::static_pointer_cast<octave_value>(value);
}

inline Eval::Value C(const octave_value &value) {
  return std::make_shared<octave_value>(value);
}

std::string OctEval::cast_string(const Eval::Value &value) const {
  if(valueIsOfType(value, StringType))
    return C(value)->string_value();
  throw std::runtime_error("Cannot cast this value to string.");
}

double OctEval::cast_double(const Eval::Value &value) const {
  if(valueIsOfType(value, ScalarType))
    return C(value)->double_value();
  throw std::runtime_error("Cannot cast this value to double.");
}

std::vector<double> OctEval::cast_vector_double(const Eval::Value &value) const {
  if(valueIsOfType(value, ScalarType))
    return std::vector<double>(1, cast<double>(value));
  else if(valueIsOfType(value, VectorType)) {
    Matrix m=C(value)->matrix_value();
    std::vector<double> ret(m.rows());
    for(int i=0; i<m.rows(); ++i)
      ret[i]=m(i, 0);
    return ret;
  }
  throw std::runtime_error("Cannot cast this value to vector<double>.");
}

std::vector<std::vector<double> > OctEval::cast_vector_vector_double(const Eval::Value &value) const {
  if(valueIsOfType(value, ScalarType))
    return std::vector<std::vector<double> >(1, std::vector<double>(1, cast<double>(value)));
  if(valueIsOfType(value, VectorType)) {
    Matrix m=C(value)->matrix_value();
    std::vector<std::vector<double> > ret(m.rows(), std::vector<double>(1));
    for(int i=0; i<m.rows(); ++i)
      ret[i][0]=m(i,0);
    return ret;
  }
  else if(valueIsOfType(value, MatrixType)) {
    Matrix m=C(value)->matrix_value();
    std::vector<std::vector<double> > ret(m.rows(), std::vector<double>(m.cols()));
    for(int r=0; r<m.rows(); ++r)
      for(int c=0; c<m.cols(); ++c)
        ret[r][c]=m(r, c);
    return ret;
  }
  throw std::runtime_error("Cannot cast this value to vector<vector<double> >.");
}

Eval::Value OctEval::create_double(const double& v) const {
  return std::make_shared<octave_value>(v);
}

Eval::Value OctEval::create_vector_double(const std::vector<double>& v) const {
  Matrix m(v.size(), 1);
  for(int i=0; i<v.size(); ++i)
    m(i)=v[i];
  return std::make_shared<octave_value>(m);
}

Eval::Value OctEval::create_vector_vector_double(const std::vector<std::vector<double> >& v) const {
  Matrix m(v.size(), v[0].size());
  for(int r=0; r<v.size(); ++r)
    for(int c=0; c<v[r].size(); ++c)
      m(c*m.rows()+r)=v[r][c];
  return std::make_shared<octave_value>(m);
}

Eval::Value OctEval::create_string(const std::string& v) const {
  return std::make_shared<octave_value>(v);
}

Eval::Value OctEval::createFunctionDep(const std::vector<Value>& v) const {
  auto ret=createSwigByTypeName("VectorSym");
  auto vec=static_cast<fmatvec::Vector<fmatvec::Var, fmatvec::SymbolicExpression>*>(getSwigPtr(*C(ret)));
  vec->resize(v.size());
  for(int i=0; i<v.size(); ++i) {
    if(valueIsOfType(v[i], ScalarType))
      (*vec)(i)=C(v[i])->double_value();
    else {
	    std::string type=getSwigType(*C(v[i]));
      if(type!="SymbolicExpression" && type!="IndependentVariable")
        throw std::runtime_error("Value is not scalar symbolic or independent variable (but is " + type + ").");
      (*vec)(i)=*static_cast<fmatvec::SymbolicExpression*>(getSwigPtr(*C(v[i])));
    }
  }
  return ret;
}

Eval::Value OctEval::createFunctionDep(const std::vector<std::vector<Value> >& v) const {
  auto ret=createSwigByTypeName("MatrixSym");
  auto mat=static_cast<fmatvec::Matrix<fmatvec::General, fmatvec::Var, fmatvec::Var, fmatvec::SymbolicExpression>*>(getSwigPtr(*C(ret)));
  mat->resize(v.size(), v[0].size());
  for(int r=0; r<v.size(); ++r)
    for(int c=0; c<v[r].size(); ++c)
      if(valueIsOfType(v[r][c], ScalarType))
        (*mat)(r,c)=C(v[r][c])->double_value();
      else {
	      std::string type=getSwigType(*C(v[r][c]));
        if(type!="SymbolicExpression" && type!="IndependentVariable")
          throw std::runtime_error("Value is not scalar symbolic or independent variable (but is " + type + ").");
        (*mat)(r,c)=*static_cast<fmatvec::SymbolicExpression*>(getSwigPtr(*C(v[r][c])));
      }
  return ret;
}

Eval::Value OctEval::createFunction(const std::vector<Value> &indeps, const Value &dep) const {
  Cell c(indeps.size()+1,1);
  for(size_t i=0; i<indeps.size(); ++i)
    c(i)=*C(indeps[i]);
  c(indeps.size())=*C(dep);
  return C(c);
}

void* OctEval::getSwigPtr(const octave_value &v) {
  static octave_function *swig_this=octInit.interpreter->get_symbol_table().find_function("swig_this").function_value(); // get ones a pointer to swig_this for performance reasons
  // get the pointer: octave returns a 64bit integer which represents the pointer
  if(v.class_name()!="swig_ref")
    throw std::runtime_error("This value is not a reference to a SWIG wrapped object.");
  octave_value swigThis=fevalThrow(swig_this, v, 1, "Cannot get pointer to the SWIG wrapped object.")(0);
  return reinterpret_cast<void*>(swigThis.uint64_scalar_value().value());
}

Eval::Value OctEval::createSwigByTypeName(const std::string &name) {
  std::list<octave_value_list> idx;
  idx.emplace_back(name);
  idx.emplace_back();
  return C(fevalThrow(octInit.interpreter->get_symbol_table().find_function("new_"+name).function_value(), octave_value_list(), 1,
    "Failed to create "+name)(0));
}

std::string OctEval::getSwigType(const octave_value &v) {
  if(v.class_name()!="swig_ref")
    return "";
  // get the swig type (get ones a pointer to swig_type for performance reasons)
  static octave_function *swig_type=octInit.interpreter->get_symbol_table().find_function("swig_type").function_value();
  return fevalThrow(swig_type, v, 1, "Unable to get swig type.")(0).string_value();
}


std::string OctEval::serializeFunction(const Value &x) const {
  auto c=C(x)->cell_value();
  int nrIndeps=c.dims()(0)-1;
  std::stringstream str;
  str<<"f(";
  for(int i=0; i<nrIndeps; ++i) {
    std::string type=getSwigType(c(i));
    if(type=="IndependentVariable")
      str<<(i==0?"":",")<<*static_cast<fmatvec::IndependentVariable*>(getSwigPtr(c(i)));
    else if(type=="VectorIndep")
      str<<(i==0?"":",")<<*static_cast<fmatvec::Vector<fmatvec::Var, fmatvec::IndependentVariable>*>(getSwigPtr(c(i)));
    else
      throw std::runtime_error("Unknown type for independent variable in function: "+type);
  }
  str<<")=";
  std::string type=getSwigType(c(nrIndeps));
  auto cc=C(c(nrIndeps));
  if(valueIsOfType(cc, ScalarType))
    str<<fmatvec::SymbolicExpression(cast<double>(cc));
  else if(valueIsOfType(cc, VectorType))
    str<<static_cast<fmatvec::Vector<fmatvec::Var, fmatvec::SymbolicExpression>>(
              fmatvec::VecV(cast<std::vector<double>>(cc)));
  else if(valueIsOfType(cc, MatrixType))
    str<<static_cast<fmatvec::Matrix<fmatvec::General, fmatvec::Var, fmatvec::Var, fmatvec::SymbolicExpression>>(
              fmatvec::MatV(cast<std::vector<std::vector<double>>>(cc)));
  else if(type=="SymbolicExpression" || type=="IndependentVariable")
    str<<*static_cast<fmatvec::SymbolicExpression*>(getSwigPtr(c(nrIndeps)));
  else if(type=="VectorSym" || type=="VectorIndep")
    str<<*static_cast<fmatvec::Vector<fmatvec::Var, fmatvec::SymbolicExpression>*>(getSwigPtr(c(nrIndeps)));
  else if(type=="MatrixSym")
    str<<*static_cast<fmatvec::Matrix<fmatvec::General, fmatvec::Var, fmatvec::Var, fmatvec::SymbolicExpression>*>(getSwigPtr(c(nrIndeps)));
  else
    throw std::runtime_error("Unknown type for dependent variable in function: "+type);
  return str.str();
}

OctEval::OctEval(std::vector<bfs::path> *dependencies_) : Eval(dependencies_) {
  auto ci=std::make_shared<Import>();
  currentImport=ci;
  ci->path=octInit.initialPath;

  // add .../share/mbxmlutils/octave to octave search path (MBXMLUtils m-files are stored their)
  addImportHelper(installPath/"share"/"mbxmlutils"/"octave");
};

OctEval::~OctEval() = default;

Eval::Value OctEval::createFunctionIndep(int dim) const {
  if(dim==0)
    return createSwigByTypeName("IndependentVariable");
  auto ret=createSwigByTypeName("VectorIndep");
  auto vec=static_cast<fmatvec::Vector<fmatvec::Var, fmatvec::IndependentVariable>*>(getSwigPtr(*C(ret)));
  vec->resize(dim, fmatvec::NONINIT);
  return ret;
}

void OctEval::addImportHelper(const boost::filesystem::path &dir) {
  // some special handing for the octave addpath is required since addpath is very time consuming
  // in octave. Hence we try to change the path as less as possible. See also fullStringToValue.

  static octave_function *addpath=octInit.interpreter->get_symbol_table().find_function("addpath").function_value(); // get ones a pointer for performance reasons
  static octave_function *path=octInit.interpreter->get_symbol_table().find_function("path").function_value(); // get ones a pointer for performance reasons
  // set octave path to top of stack of not already done
  std::string curPath;
  curPath=fevalThrow(path, octave_value_list(), 1)(0).string_value();
  auto ci=std::static_pointer_cast<Import>(currentImport);
  std::string &currentPath=ci->path;
  if(curPath!=currentPath)
  {
    // set path
    fevalThrow(path, octave_value_list(octave_value(currentPath)), 0,
      "Unable to set the octave search path "+currentPath);
  }
  // add dir to octave path
#if OCTAVE_MAJOR_VERSION >= 6
  auto vnBefore=octInit.interpreter->variable_names();
  auto gvnBefore=octInit.interpreter->global_variable_names();
  auto ufnBefore=octInit.interpreter->get_symbol_table().user_function_names();
  auto tlvnBefore=octInit.interpreter->top_level_variable_names();
#else
  auto vnBefore=octInit.interpreter->get_symbol_table().variable_names();
#endif
  fevalThrow(addpath, octave_value_list(octave_value(absolute(dir).string())), 0,
    "Unable to add octave search path "+absolute(dir).string());
#if OCTAVE_MAJOR_VERSION >= 6
  auto vnAfter=octInit.interpreter->variable_names();
  auto gvnAfter=octInit.interpreter->global_variable_names();
  auto ufnAfter=octInit.interpreter->get_symbol_table().user_function_names();
  auto tlvnAfter=octInit.interpreter->top_level_variable_names();
#else
  auto vnAfter=octInit.interpreter->get_symbol_table().variable_names();
#endif
  // get new path and store it in top of stack
  currentPath=fevalThrow(path, octave_value_list(), 1)(0).string_value();

  // create a list of all variables added by the addPath command and register these as parameter
  // to restore it in any new context with this addPath.
  auto fillVars=[](const std::list<std::string> &listBefore, const std::list<std::string> &listAfter, std::map<std::string, octave_value> &im,
                   const std::function<octave_value(const std::string&)> &get){
    std::set<std::string> setBefore(listBefore.begin(), listBefore.end());
    std::set<std::string> setAfter(listAfter.begin(), listAfter.end());
    std::set<std::string> newVars;
    set_difference(setAfter.begin(), setAfter.end(), setBefore.begin(), setBefore.end(), inserter(newVars, newVars.begin()));
    for(auto &n : newVars)
      im[n]=get(n);
  };
#if OCTAVE_MAJOR_VERSION >= 6
  auto *octIntPtr=octInit.interpreter;
  auto *gstPtr=&octInit.interpreter->get_symbol_table();
  fillVars(vnBefore  , vnAfter  , ci->vn  , [octIntPtr](auto && name) { return octIntPtr->varval(name); });
  fillVars(gvnBefore , gvnAfter , ci->gvn , [octIntPtr](auto && name) { return octIntPtr->global_varval(name); });
  fillVars(ufnBefore , ufnAfter , ci->ufn , [gstPtr]   (auto && name) { return gstPtr->find_user_function(name); });
  fillVars(tlvnBefore, tlvnAfter, ci->tlvn, [octIntPtr](auto && name) { return octIntPtr->top_level_varval(name); });
#else
  auto gstPtr=&octInit.interpreter->get_symbol_table();
  fillVars(vnBefore  , vnAfter  , ci->vn  , bind(&octave::symbol_table::varval                 , gstPtr   , std::placeholders::_1));
#endif
}

void OctEval::addImport(const std::string &code, const DOMElement *e) {
  try {
    bfs::path dir;
    if(e) {
      // evaluate code to and string (directory to add using addpath)
      dir=cast<std::string>(fullStringToValue(code, e));
      // convert to an absolute path using e
      dir=E(e)->convertPath(dir);
    }
    else
      dir=code;

    addImportHelper(dir);

    if(!dependencies)
      return;
    // add m-files in dir to dependencies
    for(bfs::directory_iterator it=bfs::directory_iterator(dir); it!=bfs::directory_iterator(); it++)
      if(it->path().extension()==".m")
        dependencies->push_back(it->path());
  } RETHROW_AS_DOMEVALEXCEPTION(e)
}

Eval::Value OctEval::fullStringToValue(const std::string &str, const DOMElement *e, bool skipRet) const {
  // check some common string to avoid time consiming evaluation
  // check true and false
  if(str=="true") return std::make_shared<octave_value>(1);
  if(str=="false") return std::make_shared<octave_value>(0);
  // check for floating point values
  double d;
  char *end;
  d=strtod(str.c_str(), &end);
  if(end!=str && boost::algorithm::trim_copy(std::string(end))=="")
    return std::make_shared<octave_value>(d);
  // no common string detected -> evaluate using octave now

  // restore current dir on exit and change current dir
  PreserveCurrentDir preserveDir;
  if(e) {
    bfs::path chdir=E(e)->getOriginalFilename().parent_path();
    if(!chdir.empty())
      bfs::current_path(chdir);
  }

  // clear local octave variables
  // octInit.interpreter->get_symbol_table().current_scope().clear_variables() seems to be buggy, it sometimes clears
  // also global variables. Hence, we clear all variables which are not also part of the global variables
#if OCTAVE_MAJOR_VERSION >= 6
  auto gvn=octInit.interpreter->global_variable_names(); // get global and all variable names
  auto vn=octInit.interpreter->variable_names();
#else
  auto gvn=octInit.interpreter->get_symbol_table().global_variable_names(); // get global and all variable names
  auto vn=octInit.interpreter->get_symbol_table().variable_names();
#endif
  gvn.sort(); // sort global and all variable names
  vn.sort();
  std::list<std::string> lvn;
  set_difference(vn.begin(), vn.end(), gvn.begin(), gvn.end(), inserter(lvn, lvn.begin())); // get local = all - global variable names
  for(auto &l : lvn) // remove all local variable names
#if OCTAVE_MAJOR_VERSION >= 6
    octInit.interpreter->clear_symbol(l);
#else
    octInit.interpreter->get_symbol_table().clear_symbol(l);
#endif

  // restore current parameters
  for(const auto & i : currentParam)
#if OCTAVE_MAJOR_VERSION >= 6
    octInit.interpreter->assign(i.first, *C(i.second));
#else
    octInit.interpreter->get_symbol_table().assign(i.first, *C(i.second));
#endif

  // change the octave serach path only if required (for performance reasons; addpath/path(...) is very time consuming, but not path())
  static octave_function *path=octInit.interpreter->get_symbol_table().find_function("path").function_value(); // get ones a pointer for performance reasons
  std::string curPath;
  try { curPath=fevalThrow(path, octave_value_list(), 1)(0).string_value(); } RETHROW_AS_DOMEVALEXCEPTION(e)
  auto ci=std::static_pointer_cast<Import>(currentImport);
  std::string &currentPath=ci->path;
  if(curPath!=currentPath)
  {
    // set path
    try { fevalThrow(path, octave_value_list(octave_value(currentPath)), 0,
      "Unable to set the octave search path "+currentPath); } RETHROW_AS_DOMEVALEXCEPTION(e)
  }

  // restore variables from import
  auto restoreVars=[](std::map<std::string, octave_value> &im, const std::function<void(const std::string&, const octave_value &v)> &set) {
    for(auto &i : im)
      set(i.first, i.second);
  };
#if OCTAVE_MAJOR_VERSION >= 6
  auto *octIntPtr=octInit.interpreter;
  auto *gstPtr=&octInit.interpreter->get_symbol_table();
  restoreVars(ci->vn  , [octIntPtr](auto && name, auto && value) { return octIntPtr->assign(name, value); });
  restoreVars(ci->gvn , [octIntPtr](auto && name, auto && value) { return octIntPtr->global_assign(name, value); });
  restoreVars(ci->ufn , [gstPtr]   (auto && name, auto && value) { return gstPtr->install_user_function(name, value); });
  restoreVars(ci->tlvn, [octIntPtr](auto && name, auto && value) { return octIntPtr->top_level_assign(name, value); });
#else
  auto gstPtr=&octInit.interpreter->get_symbol_table();
  using GstFuncType = void(octave::symbol_table::*)(const std::string&, const octave_value &);
  restoreVars(ci->vn  , std::bind<GstFuncType>(&octave::symbol_table::assign    , gstPtr   , std::placeholders::_1, std::placeholders::_2));
#endif

  std::ostringstream err;
  std::ostringstream out;
  try {
    int dummy;
    REDIR_STDOUT(out.rdbuf());
    REDIR_STDERR(err.rdbuf());
    mbxmlutilsStaticDependencies.clear();
    if(e)
      originalFilename=E(e)->getOriginalFilename();
    else
      originalFilename.clear();
#if OCTAVE_MAJOR_VERSION >= 5
    octInit.interpreter->eval_string(str, true, dummy, 0); // eval as statement list
#else
    octave::eval_string(str, true, dummy, 0); // eval as statement list
#endif
    addStaticDependencies(e);
  }
  catch(octave::execution_exception &ex)
  {
    // octave::execution_exception is not derived (in older release) from std::exception -> convert it to an std::exception.
    throw DOMEvalException("Caught octave exception " + ex.
#if OCTAVE_MAJOR_VERSION >= 6
      message()
#else
      info()
#endif
      + "\n" +out.str() + "\n" + err.str(), e);
  }
  catch(octave::exit_exception &ex)
  {
    // octave::exit_exception is not derived (in older release) from std::exception -> convert it to an std::exception.
    throw DOMEvalException("Caught octave exception (Exit exception)\n" +out.str() + "\n" + err.str(), e);
  }
  catch(octave::interrupt_exception &ex)
  {
    // octave::interrupt_exception is not derived (in older release) from std::exception -> convert it to an std::exception.
    throw DOMEvalException("Caught octave exception (Interrupt exception)\n" +out.str() + "\n" + err.str(), e);
  }
  catch(const std::exception &ex) { // should not happend
    throw DOMEvalException(std::string(ex.what())+":\n"+out.str()+"\n"+err.str(), e);
  }
  catch(...) { // should not happend
    throw DOMEvalException("Unknwon exception:\n"+out.str()+"\n"+err.str(), e);
  }
  // generate a strNoSpace from str by removing leading/trailing spaces as well as trailing ';'.
  std::string strNoSpace=str;
  while(!strNoSpace.empty() && (strNoSpace[0]==' ' || strNoSpace[0]=='\n'))
    strNoSpace=strNoSpace.substr(1);
  while(!strNoSpace.empty() && (strNoSpace[strNoSpace.size()-1]==' ' || strNoSpace[strNoSpace.size()-1]==';' ||
    strNoSpace[strNoSpace.size()-1]=='\n'))
    strNoSpace=strNoSpace.substr(0, strNoSpace.size()-1);
#if OCTAVE_MAJOR_VERSION >= 5
  if(!octInit.interpreter->is_variable("ret") &&
     !octInit.interpreter->is_variable("ans") &&
     !octInit.interpreter->is_variable(strNoSpace)) {
#else
  if(!octInit.interpreter->get_symbol_table().current_scope().is_variable("ret") &&
     !octInit.interpreter->get_symbol_table().current_scope().is_variable("ans") &&
     !octInit.interpreter->get_symbol_table().current_scope().is_variable(strNoSpace)) {
#endif
    throw DOMEvalException("'ret' variable not defined in multi statement octave expression or incorrect single statement: "+
      str, e);
  }
  if(skipRet)
    return {};
  octave_value ret;
#if OCTAVE_MAJOR_VERSION >= 5
  if(octInit.interpreter->is_variable(strNoSpace))
    ret=octInit.interpreter->varval(strNoSpace);
  else if(!octInit.interpreter->is_variable("ret"))
    ret=octInit.interpreter->varval("ans");
  else
    ret=octInit.interpreter->varval("ret");
#else
  if(octInit.interpreter->get_symbol_table().current_scope().is_variable(strNoSpace))
    ret=octInit.interpreter->get_symbol_table().varval(strNoSpace);
  else if(!octInit.interpreter->get_symbol_table().current_scope().is_variable("ret"))
    ret=octInit.interpreter->get_symbol_table().varval("ans");
  else
    ret=octInit.interpreter->get_symbol_table().varval("ret");
#endif

  return C(ret);
}

bool OctEval::valueIsOfType(const Value &value, OctEval::ValueType type) const {
  std::shared_ptr<octave_value> v=C(value);
  switch(type) {
    case ScalarType:
      if(!v->is_string() && v->is_scalar_type() && v->isreal()) return true;
      return false;

    case VectorType:
      if(valueIsOfType(value, ScalarType)) return true;
      if(!v->is_string() && v->is_matrix_type() && v->isreal()) {
        Matrix m=v->matrix_value();
        if(m.cols()==1 || (m.cols()==0 && m.rows()==0)) return true; // a 0x0 matrix = [] is also treated as a vector (of size 0)
      }
      return false;

    case MatrixType:
      if(valueIsOfType(value, ScalarType)) return true;
      if(valueIsOfType(value, VectorType)) return true;
      if(!v->is_string() && v->is_matrix_type() && v->isreal()) return true;
      return false;

    case StringType:
      if(v->is_string()) return true;
      return false;

    case FunctionType:
      if(v->iscell()) return true;
      return false;
  }
  return false;
}

octave_value_list OctEval::fevalThrow(octave_function *func, const octave_value_list &arg, int n,
                                       const std::string &msg) {
  std::ostringstream out;
  std::ostringstream err;
  octave_value_list ret;
  {
    try
    {
      REDIR_STDOUT(out.rdbuf());
      REDIR_STDERR(err.rdbuf());
      ret=octave::feval(func, arg, n);
    }
    catch(octave::execution_exception &ex)
    {
      // octave::execution_exception is not derived (in older release) from std::exception -> convert it to an std::exception.
      throw std::runtime_error( "Caught octave exception " + ex.
#if OCTAVE_MAJOR_VERSION >= 6
        message()
#else
        info()
#endif
        + "\n" +out.str() + "\n" + err.str() );
    }
    catch(octave::exit_exception &ex)
    {
      // octave::exit_exception is not derived (in older release) from std::exception -> convert it to an std::exception.
      throw std::runtime_error( "Caught octave exception (Exit exception)\n" +out.str() + "\n" + err.str() );
    }
    catch(octave::interrupt_exception &ex)
    {
      // octave::interrupt_exception is not derived (in older release) from std::exception -> convert it to an std::exception.
      throw std::runtime_error( "Caught octave exception (Interrupt exception)\n" +out.str() + "\n" + err.str() );
    }
  }
  return ret;
}

std::map<bfs::path, std::pair<bfs::path, bool> >& OctEval::requiredFiles() const {
  static std::map<bfs::path, std::pair<bfs::path, bool> > files;
  if(!files.empty())
    return files;

  fmatvec::Atom::msgStatic(Info)<<"Generate file list for MBXMLUtils m-files."<<std::endl;
  for(bfs::directory_iterator srcIt=bfs::directory_iterator(installPath/"share"/"mbxmlutils"/"octave");
    srcIt!=bfs::directory_iterator(); ++srcIt) {
    if(is_directory(*srcIt)) // skip directories
      continue;
    files[srcIt->path()]=std::make_pair(bfs::path("share")/"mbxmlutils"/"octave", false);
  }

  files[installPath/LIBDIR/"fmatvec_symbolic_swig_octave.oct"]=std::make_pair(LIBDIR, true);
  files[installPath/LIBDIR/"registerPath.oct"]=std::make_pair(LIBDIR, true);

  // get octave fcnfiledir without octave_prefix
  bfs::path octave_fcnfiledir(octave::config::fcn_file_dir().substr(octave::config::octave_home().length()+1));
  fmatvec::Atom::msgStatic(Info)<<"Generate file list for octave m-files."<<std::endl;
  bfs::path dir=octave::config::octave_home()/octave_fcnfiledir;
  size_t depth=std::distance(dir.begin(), dir.end());
  for(bfs::recursive_directory_iterator srcIt=bfs::recursive_directory_iterator(dir);
      srcIt!=bfs::recursive_directory_iterator(); ++srcIt) {
    if(is_directory(*srcIt)) // skip directories
      continue;
    bfs::path::iterator dstIt=srcIt->path().begin();
    for(int i=0; i<depth; ++i) ++dstIt;
    bfs::path dst;
    for(; dstIt!=--srcIt->path().end(); ++dstIt)
      dst/=*dstIt;
    files[srcIt->path()]=std::make_pair(octave_fcnfiledir/dst, false);
  }

  // get octave octfiledir without octave_prefix
  bfs::path octave_octfiledir(octave::config::oct_file_dir().substr(octave::config::octave_home().length()+1));
  fmatvec::Atom::msgStatic(Info)<<"Generate file list for octave oct-files (excluding dependencies)."<<std::endl;
  for(bfs::directory_iterator srcIt=bfs::directory_iterator(octave::config::octave_home()/octave_octfiledir);
      srcIt!=bfs::directory_iterator(); ++srcIt) {
    if(srcIt->path().filename()=="__init_gnuplot__.oct") continue;
    if(srcIt->path().extension()==".oct")
      files[srcIt->path()]=std::make_pair(octave_octfiledir, true);
    if(srcIt->path().filename()=="PKG_ADD")
      files[srcIt->path()]=std::make_pair(octave_octfiledir, false);
  }

  return files;
}

Eval::Value OctEval::callFunction(const std::string &name, const std::vector<Value>& args) const {
  static std::map<std::string, octave_function*> functionValue;
  std::pair<std::map<std::string, octave_function*>::iterator, bool> f=functionValue.insert(std::make_pair(name, static_cast<octave_function*>(nullptr)));
  if(f.second)
    f.first->second=octInit.interpreter->get_symbol_table().find_function(name).function_value(); // get ones a pointer performance reasons
  octave_value_list octargs;
  for(const auto & arg : args)
    octargs.append(*C(arg));
  octave_value_list ret=fevalThrow(f.first->second, octargs, 1,
    "Unable to call function "+name+".");
  return C(ret(0));
}

} // end namespace MBXMLUtils
