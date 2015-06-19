#include <config.h>
#include "mbxmlutils/octeval.h"
#include <octave/version.h> // we need to check for the octave version (octave interface change)
#include <stdexcept>
#include <boost/filesystem.hpp>
#include <boost/locale.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/lexical_cast.hpp>
#include <octave/octave.h>
#include <octave/defaults.h>
#include <octave/toplev.h>
#include <mbxmlutilshelper/dom.h>
#include <mbxmlutilshelper/utils.h>
#include <mbxmlutilshelper/getinstallpath.h>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMAttr.hpp>

//TODO: should also compile if casadi is not present

using namespace std;
using namespace xercesc;
using namespace boost;
namespace bfs=boost::filesystem;

namespace {
  //TODO not working on Windows
  //TODO // NOTE: we can skip the use of utf8Facet (see below) and set the facet globally (for bfs::path and others) using:
  //TODO // std::locale::global(boost::locale::generator().generate("UTF8"));
  //TODO // boost::filesystem::path::imbue(std::locale());
  //TODO const bfs::path::codecvt_type *utf8Facet(&use_facet<bfs::path::codecvt_type>(boost::locale::generator().generate("UTF8")));
  #define CODECVT
}

namespace MBXMLUtils {

// Store the current directory in the ctor an restore in the dtor
class PreserveCurrentDir {
  public:
    PreserveCurrentDir() {
      dir=bfs::current_path();
    }
    ~PreserveCurrentDir() {
      bfs::current_path(dir);
    }
  private:
    bfs::path dir;
};

string OctEval::cast_string(const shared_ptr<void> &value) {
  static const double eps=pow(10, -numeric_limits<double>::digits10-2);
  ValueType type=getType(value);
  ostringstream ret;
  ret.precision(numeric_limits<double>::digits10+1);
  if(type==StringType) {
    ret<<"'"<<C(value)->string_value()<<"'";
    return ret.str();
  }
  if(type==ScalarType || type==VectorType || type==MatrixType || type==SXType || type==DMatrixType) {
    vector<vector<double> > m=cast<vector<vector<double> > >(value);
    if(type!=ScalarType)
      ret<<"[";
    for(int i=0; i<m.size(); i++) {
      for(int j=0; j<m[i].size(); j++) {
        int mint=0;
        try {
           mint=boost::math::lround(m[i][j]);
        }
        catch(...) {}
        double delta=fabs(mint-m[i][j]);
        if(delta>eps*m[i][j] && delta>eps)
          ret<<m[i][j]<<(j<m[i].size()-1?",":"");
        else
          ret<<mint<<(j<m[i].size()-1?",":"");
      }
      ret<<(i<m.size()-1?" ; ":"");
    }
    if(type!=ScalarType || m.size()==0)
      ret<<"]";
    return ret.str();
  }
  throw DOMEvalException("Cannot cast this value to a string.");
}

int OctEval::cast_int(const shared_ptr<void> &value) {
  static const double eps=pow(10, -numeric_limits<double>::digits10-2);
  vector<vector<double> > m=cast<vector<vector<double> > >(value);
  if(getType(value)==ScalarType) {
    int ret=boost::math::lround(m[0][0]);
    double delta=fabs(ret-m[0][0]);
    if(delta>eps*m[0][0] && delta>eps)
      throw DOMEvalException("Canot cast this value to int.");
    else
      return ret;
  }
  throw DOMEvalException("Cannot cast this value to int.");
}

double OctEval::cast_double(const shared_ptr<void> &value) {
  vector<vector<double> > m=cast<vector<vector<double> > >(value);
  if(getType(value)==ScalarType)
    return m[0][0];
  throw DOMEvalException("Cannot cast this value to double.");
}

vector<double> OctEval::cast_vector_double(const shared_ptr<void> &value) {
  vector<vector<double> > m=cast<vector<vector<double> > >(value);
  ValueType type=getType(value);
  if(type==ScalarType || type==VectorType) {
    vector<double> ret;
    for(size_t i=0; i<m.size(); i++)
      ret.push_back(m[i][0]);
    return ret;
  }
  throw DOMEvalException("Cannot cast this value to vector<double>.");
}

vector<vector<double> > OctEval::cast_vector_vector_double(const shared_ptr<void> &value) {
  ValueType type=getType(value);
  if(type==ScalarType || type==VectorType || type==MatrixType) {
    vector<vector<double> > ret;
    Matrix m=C(value)->matrix_value();
    for(int i=0; i<m.rows(); i++) {
      vector<double> row;
      for(int j=0; j<m.cols(); j++)
        row.push_back(m(j*m.rows()+i));
      ret.push_back(row);
    }
    return ret;
  }
  if(type==SXType || type==DMatrixType) {
    casadi::SX m;
    if(type==SXType)
      m=cast<casadi::SX>(value);
    else
      m=cast<casadi::DMatrix>(value);
    if(!m.isConstant())
      throw DOMEvalException("Can only cast this constant value to vector<vector<double> >.");
    vector<vector<double> > ret;
    for(int i=0; i<m.size1(); i++) {
      vector<double> row;
      for(int j=0; j<m.size2(); j++)
        row.push_back(m.elem(i,j).getValue());
      ret.push_back(row);
    }
    return ret;
  }
  throw DOMEvalException("Cannot cast this value to vector<vector<double> >.");
}

DOMElement* OctEval::cast_DOMElement_p(const shared_ptr<void> &value, DOMDocument *doc) {
  if(getType(value)==SXFunctionType)
    return convertCasADiToXML(cast<casadi::SXFunction>(value), doc);
  throw DOMEvalException("Cannot cast this value to DOMElement*.");
}

casadi::SX OctEval::cast_SX(const shared_ptr<void> &value) {
  ValueType type=getType(value);
  if(type==SXType)
    return Ptr<casadi::SX>::cast(castToSwig(value));
  if(type==DMatrixType)
    return Ptr<casadi::DMatrix>::cast(castToSwig(value));
  if(type==ScalarType || type==VectorType || type==MatrixType) {
    vector<vector<double> > m=cast<vector<vector<double> > >(value);
    casadi::SX ret=casadi::SX::zeros(m.size(), m[0].size());
    for(int i=0; i<m.size(); i++)
      for(int j=0; j<m[i].size(); j++)
        ret.elem(i,j)=m[i][j];
    return ret;
  }
  throw DOMEvalException("Cannot cast this value to casadi::SX.");
}

casadi::SX* OctEval::cast_SX_p(const shared_ptr<void> &value) {
  if(getType(value)==SXType)
    return Ptr<casadi::SX*>::cast(castToSwig(value));
  throw DOMEvalException("Cannot cast this value to casadi::SX*.");
}

casadi::DMatrix OctEval::cast_DMatrix(const shared_ptr<void> &value) {
  ValueType type=getType(value);
  if(type==DMatrixType)
    return Ptr<casadi::DMatrix>::cast(castToSwig(value));
  if(type==ScalarType || type==VectorType || type==MatrixType || type==SXType) {
    vector<vector<double> > m=cast<vector<vector<double> > >(value);
    casadi::DMatrix ret=casadi::DMatrix::zeros(m.size(), m[0].size());
    for(int i=0; i<m.size(); i++)
      for(int j=0; j<m[i].size(); j++)
        ret.elem(i,j)=m[i][j];
    return ret;
  }
  throw DOMEvalException("Cannot cast this value to casadi::DMatrix.");
}

casadi::DMatrix* OctEval::cast_DMatrix_p(const shared_ptr<void> &value) {
  if(getType(value)==DMatrixType)
    return Ptr<casadi::DMatrix*>::cast(castToSwig(value));
  throw DOMEvalException("Cannot cast this value to casadi::DMatrix*.");
}

casadi::SXFunction OctEval::cast_SXFunction(const shared_ptr<void> &value) {
  if(getType(value)==SXFunctionType)
    return Ptr<casadi::SXFunction>::cast(castToSwig(value));
  throw DOMEvalException("Cannot cast this value to casadi::SXFunction.");
}

casadi::SXFunction* OctEval::cast_SXFunction_p(const shared_ptr<void> &value) {
  if(getType(value)==SXFunctionType)
    return Ptr<casadi::SXFunction*>::cast(castToSwig(value));
  throw DOMEvalException("Cannot cast this value to casadi::SXFunction*.");
}

octave_value OctEval::createCasADi(const string &name) {
  list<octave_value_list> idx;
  idx.push_back(octave_value_list(name));
  idx.push_back(octave_value_list());
  return casadiOctValue->subsref(".(", idx);
}

int OctEval::initCount=0;
string OctEval::initialPath;
string OctEval::pathSep;

std::map<std::string, std::string> OctEval::units;

boost::scoped_ptr<octave_value> OctEval::casadiOctValue;

OctEval::OctEval(vector<bfs::path> *dependencies_) : Eval(dependencies_) {
  initCount++;
  if(initCount==1) {
    try {
      // set the OCTAVE_HOME envvar and octave_prefix variable before initializing octave
      bfs::path octave_prefix(OCTAVE_PREFIX); // hard coded default (setting OCTAVE_HOME not requried)
      if(getenv("OCTAVE_HOME")) // OCTAVE_HOME set manually -> use this for octave_prefix
        octave_prefix=getenv("OCTAVE_HOME");
      else if(getenv("OCTAVE_HOME")==NULL && bfs::exists(MBXMLUtils::getInstallPath()/"share"/"octave")) {
        // OCTAVE_HOME not set but octave is available in installation path of MBXMLUtils -> use installation path
        octave_prefix=MBXMLUtils::getInstallPath();
        // the string for putenv must have program life time
        static string OCTAVE_HOME="OCTAVE_HOME="+MBXMLUtils::getInstallPath().string(CODECVT);
        putenv((char*)OCTAVE_HOME.c_str());
      }

      // initialize octave
      static vector<char*> octave_argv;
      octave_argv.resize(6);
      octave_argv[0]=const_cast<char*>("embedded");
      octave_argv[1]=const_cast<char*>("--no-history");
      octave_argv[2]=const_cast<char*>("--no-init-file");
      octave_argv[3]=const_cast<char*>("--no-line-editing");
      octave_argv[4]=const_cast<char*>("--no-window-system");
      octave_argv[5]=const_cast<char*>("--silent");
      octave_main(6, &octave_argv[0], 1);
  
      // set some global octave config
      octave_value_list warnArg;
      warnArg.append("error");
      warnArg.append("Octave:divide-by-zero");
      feval("warning", warnArg);
      if(error_state!=0) { error_state=0; throw runtime_error("Internal error: unable to disable warnings."); }

      if(pathSep.empty()) {
        octave_value ret=fevalThrow(symbol_table::find_function("pathsep").function_value(), octave_value_list(), 1,
          "Internal error: Unable to get the path seperator")(0);
        pathSep=ret.string_value();
      }

      // remove the default oct serach path ...
      // (first get octave octfiledir without octave_prefix)
      bfs::path octave_octfiledir(string(OCTAVE_OCTFILEDIR).substr(string(OCTAVE_PREFIX).length()+1));
      { // print no warning, path may not exist
        BLOCK_STDERR(blockstderr);
        feval("rmpath", octave_value_list(octave_value((octave_prefix/octave_octfiledir).string())));
      }

      // ... and add .../[bin|lib] to octave search path (their we push all oct files)
      string dir=(MBXMLUtils::getInstallPath()/"lib").string(CODECVT);
      feval("addpath", octave_value_list(octave_value(dir)));
      if(error_state!=0) { error_state=0; throw runtime_error("Internal error: cannot add octave search path."); }

      // ... and add .../bin to octave search path (their we push all oct files)
      dir=(MBXMLUtils::getInstallPath()/"bin").string(CODECVT);
      feval("addpath", octave_value_list(octave_value(dir)));
      if(error_state!=0) { error_state=0; throw runtime_error("Internal error: cannot add octave search path."); }

      // add .../share/mbxmlutils/octave to octave search path (MBXMLUtils m-files are stored their)
      dir=(MBXMLUtils::getInstallPath()/"share"/"mbxmlutils"/"octave").string(CODECVT);
      feval("addpath", octave_value_list(octave_value(dir)));
      if(error_state!=0) { error_state=0; throw runtime_error("Internal error: cannot add octave search path."); }

      // save initial octave search path
      initialPath=feval("path", octave_value_list(), 1)(0).string_value();

      // load casadi
      {
        BLOCK_STDERR(blockstderr);
        casadiOctValue.reset(new octave_value(feval("swigLocalLoad", octave_value_list("casadi_oct"), 1)(0)));
        if(error_state!=0) { error_state=0; throw runtime_error("Internal error: unable to initialize casadi."); }
      }

      // get units
      msg(Info)<<"Build unit list for measurements."<<endl;
      bfs::path XMLDIR=MBXMLUtils::getInstallPath()/"share"/"mbxmlutils"/"xml"; // use rel path if build configuration dose not work
      boost::shared_ptr<DOMDocument> mmdoc=DOMParser::create(false)->parse(XMLDIR/"measurement.xml", dependencies);
      DOMElement *ele, *el2;
      for(ele=mmdoc->getDocumentElement()->getFirstElementChild(); ele!=0; ele=ele->getNextElementSibling())
        for(el2=ele->getFirstElementChild(); el2!=0; el2=el2->getNextElementSibling()) {
          if(units.find(E(el2)->getAttribute("name"))!=units.end())
            throw runtime_error(string("Internal error: Unit name ")+E(el2)->getAttribute("name")+" is defined more than once.");
          units[E(el2)->getAttribute("name")]=X()%E(el2)->getFirstTextChild()->getData();
        }
    }
    catch(...) {
      deinitOctave();
      throw;
    }
  }
  pathStack.push(initialPath);
};

OctEval::~OctEval() {
  deinitOctave();
}

void OctEval::deinitOctave() {
  initCount--;
  if(initCount==0) {
    casadiOctValue.reset();
    //Workaround: eval a VALID dummy statement before leaving "main" to prevent a crash in post main
    int dummy;
    eval_string("1+1;", true, dummy, 0); // eval as statement list

    // cleanup ocatve, but do NOT call ::exit
    octave_exit=NULL; // do not call ::exit
    clean_up_and_exit(0);
  }
}

void OctEval::addParam(const std::string &paramName, const shared_ptr<void>& value) {
  currentParam[paramName]=value;
}

void OctEval::addParamSet(const DOMElement *e) {
  for(const DOMElement *ee=e->getFirstElementChild(); ee!=NULL; ee=ee->getNextElementSibling()) {
    if(E(ee)->getTagName()==PV%"searchPath") {
      shared_ptr<octave_value> ret=C(eval(E(ee)->getAttributeNode("href"), ee));
      try { addPath(E(ee)->convertPath(ret->string_value()), ee); } MBXMLUTILS_RETHROW(e)
    }
    else {
      shared_ptr<octave_value> ret=C(eval(ee));
      addParam(E(ee)->getAttribute("name"), ret);
    }
  }
}

void OctEval::addPath(const bfs::path &dir, const DOMElement *e) {
  static octave_function *addpath=symbol_table::find_function("addpath").function_value(); // get ones a pointer for performance reasons
  static octave_function *path=symbol_table::find_function("path").function_value(); // get ones a pointer for performance reasons
  // set octave path to top of stack of not already done
  string curPath;
  try { curPath=fevalThrow(path, octave_value_list(), 1)(0).string_value(); } MBXMLUTILS_RETHROW(e)
  if(curPath!=pathStack.top())
  {
    // set path
    try { fevalThrow(path, octave_value_list(octave_value(pathStack.top())), 0,
      "Unable to set the octave search path "+pathStack.top()); } MBXMLUTILS_RETHROW(e)
  }
  // add dir to octave path
  try { fevalThrow(addpath, octave_value_list(octave_value(absolute(dir).string())), 0,
    "Unable to add octave search path "+absolute(dir).string()); } MBXMLUTILS_RETHROW(e)
  // get new path and store it in top of stack
  try { pathStack.top()=fevalThrow(path, octave_value_list(), 1)(0).string_value(); } MBXMLUTILS_RETHROW(e)

  if(!dependencies)
    return;
  // add m-files in dir to dependencies
  for(bfs::directory_iterator it=bfs::directory_iterator(dir); it!=bfs::directory_iterator(); it++)
    if(it->path().extension()==".m")
      dependencies->push_back(it->path());
}

shared_ptr<void> OctEval::stringToValue(const string &str, const DOMElement *e, bool fullEval) {
  if(fullEval)
    return C(fullStringToOctValue(str, e));
  else
    return C(partialStringToOctValue(str, e));
}

octave_value OctEval::fullStringToOctValue(const string &str, const DOMElement *e) {
  // check some common string to avoid time consiming evaluation
  // check true and false
  if(str=="true") return 1;
  if(str=="false") return 0;
  // check for floating point values
  try { return boost::lexical_cast<double>(str); }
  catch(const boost::bad_lexical_cast &) {}
  // no common string detected -> evaluate using octave now

  // restore current dir on exit and change current dir
  PreserveCurrentDir preserveDir;
  if(e) {
    bfs::path chdir=E(e)->getOriginalFilename().parent_path();
    if(!chdir.empty())
      bfs::current_path(chdir);
  }

  // clear octave variables
  symbol_table::clear_variables();
  // restore current parameters
  for(map<string, shared_ptr<void> >::const_iterator i=currentParam.begin(); i!=currentParam.end(); i++)
    #if defined OCTAVE_API_VERSION_NUMBER // check for octave < 3.8: octave < 3.8 defines this macro
      symbol_table::varref(i->first)=C(i->second);
    #else // octave >= 3.8 does not define this macro but OCTAVE_[MAJOR|...]_VERSION
      symbol_table::assign(i->first, *C(i->second));
    #endif

  // change the octave serach path only if required (for performance reasons; addpath/path(...) is very time consuming, but not path())
  static octave_function *path=symbol_table::find_function("path").function_value(); // get ones a pointer for performance reasons
  string curPath;
  try { curPath=fevalThrow(path, octave_value_list(), 1)(0).string_value(); } MBXMLUTILS_RETHROW(e)
  if(curPath!=pathStack.top())
  {
    // set path
    try { fevalThrow(path, octave_value_list(octave_value(pathStack.top())), 0,
      "Unable to set the octave search path "+pathStack.top()); } MBXMLUTILS_RETHROW(e)
  }

  ostringstream err;
  try{
    int dummy;
    BLOCK_STDOUT(blockstdout);
    REDIR_STDERR(redirstderr, err.rdbuf());
    eval_string(str, true, dummy, 0); // eval as statement list
  }
  catch(const std::exception &ex) { // should not happend
    error_state=0;
    throw DOMEvalException(string("Exception: ")+ex.what()+": "+err.str(), e);
  }
  catch(...) { // should not happend
    error_state=0;
    throw DOMEvalException("Unknwon exception: "+err.str(), e);
  }
  if(error_state!=0) { // if error => wrong code => throw error
    error_state=0;
    throw DOMEvalException(err.str()+"Unable to evaluate expression: "+str, e);
  }
  // generate a strNoSpace from str by removing leading/trailing spaces as well as trailing ';'.
  string strNoSpace=str;
  while(strNoSpace.size()>0 && (strNoSpace[0]==' ' || strNoSpace[0]=='\n'))
    strNoSpace=strNoSpace.substr(1);
  while(strNoSpace.size()>0 && (strNoSpace[strNoSpace.size()-1]==' ' || strNoSpace[strNoSpace.size()-1]==';' ||
    strNoSpace[strNoSpace.size()-1]=='\n'))
    strNoSpace=strNoSpace.substr(0, strNoSpace.size()-1);
  if(!symbol_table::is_variable("ret") && !symbol_table::is_variable("ans") && !symbol_table::is_variable(strNoSpace)) {
    throw DOMEvalException("'ret' variable not defined in multi statement octave expression or incorrect single statement: "+
      str, e);
  }
  octave_value ret;
  if(symbol_table::is_variable(strNoSpace))
    ret=symbol_table::varval(strNoSpace);
  else if(!symbol_table::is_variable("ret"))
    ret=symbol_table::varval("ans");
  else
    ret=symbol_table::varval("ret");

  return ret;
}

string OctEval::partialStringToOctValue(const string &str, const DOMElement *e) {
  string s=str;
  size_t i;
  while((i=s.find('{'))!=string::npos) {
    size_t j=i;
    do {
      j=s.find('}', j+1);
      if(j==string::npos) throw DOMEvalException("no matching } found in attriubte.", e);
    }
    while(s[j-1]=='\\'); // skip } which is quoted with backslash
    string evalStr=s.substr(i+1,j-i-1);
    // remove the backlash quote from { and }
    size_t k=0;
    while((k=evalStr.find('{', k))!=string::npos) {
      if(k==0 || evalStr[k-1]!='\\') throw DOMEvalException("{ must be quoted with a backslash inside {...}.", e);
      evalStr=evalStr.substr(0, k-1)+evalStr.substr(k);
    }
    k=0;
    while((k=evalStr.find('}', k))!=string::npos) {
      if(k==0 || evalStr[k-1]!='\\') throw DOMEvalException("} must be quoted with a backslash inside {...}.", e);
      evalStr=evalStr.substr(0, k-1)+evalStr.substr(k);
    }
    
    octave_value ret=fullStringToOctValue(evalStr, e);
    string subst;
    try {
      subst=cast<string>(C(ret));
      if(getType(C(ret))==StringType)
        subst=subst.substr(1, subst.length()-2);
    } MBXMLUTILS_RETHROW(e)
    s=s.substr(0,i)+subst+s.substr(j+1);
  }
  return s;
}

shared_ptr<void> OctEval::eval(const xercesc::DOMElement *e) {
  const DOMElement *ec;

  // check if we are evaluating a symbolic function element
  bool function=false;
  DOMNamedNodeMap *attr=e->getAttributes();
  for(int i=0; i<attr->getLength(); i++) {
    DOMAttr *a=static_cast<DOMAttr*>(attr->item(i));
    // skip xml* attributes
    if((X()%a->getName()).substr(0, 3)=="xml")
      continue;
    if(A(a)->isDerivedFrom(PV%"symbolicFunctionArgNameType")) {
      function=true;
      break;
    }
  }

  // for functions add the function arguments as parameters
  NewParamLevel newParamLevel(*this, function);
  vector<casadi::SX> inputs;
  if(function) {
    addParam("casadi", C(*casadiOctValue));
    // loop over all attributes and search for arg1name, arg2name attributes
    DOMNamedNodeMap *attr=e->getAttributes();
    for(int i=0; i<attr->getLength(); i++) {
      DOMAttr *a=static_cast<DOMAttr*>(attr->item(i));
      // skip xml* attributes
      if((X()%a->getName()).substr(0, 3)=="xml")
        continue;
      // skip all attributes not of type symbolicFunctionArgNameType
      if(!A(a)->isDerivedFrom(PV%"symbolicFunctionArgNameType"))
        continue;
      string base=X()%a->getName();
      if(!E(e)->hasAttribute(base+"Dim"))
        throw DOMEvalException("Internal error: their must also be a attribute named "+base+"Dim", e);
      if(!E(e)->hasAttribute(base+"Nr"))
        throw DOMEvalException("Internal error: their must also be a attribute named "+base+"Nr", e);
      int nr=boost::lexical_cast<int>(E(e)->getAttribute(base+"Nr"));
      int dim=boost::lexical_cast<int>(E(e)->getAttribute(base+"Dim"));

      octave_value octArg=createCasADi("SX");
      casadi::SX *arg;
      try { arg=cast<casadi::SX*>(C(octArg)); } MBXMLUTILS_RETHROW(e)
      *arg=casadi::SX::sym(X()%a->getValue(), dim, 1);
      addParam(X()%a->getValue(), C(octArg));
      inputs.resize(max(nr, static_cast<int>(inputs.size()))); // fill new elements with default ctor (isNull()==true)
      inputs[nr-1]=*arg;
    }
    // check if one argument was not set. If so error
    for(int i=0; i<inputs.size(); i++)
      if(inputs[i].sparsity().isEmpty()) // a empty object is a error (see above), since not all arg?name args were defined
        throw DOMEvalException("All argXName attributes up to the largest argument number must be specified.", e);
  }
  
  // a XML vector
  ec=E(e)->getFirstElementChildNamed(PV%"xmlVector");
  if(ec) {
    int i;
    // calculate nubmer for rows
    i=0;
    for(const DOMElement* ele=ec->getFirstElementChild(); ele!=0; ele=ele->getNextElementSibling(), i++);
    // get/eval values
    Matrix m(i, 1);
    casadi::SX M;
    if(function)
      M.resize(i, 1);
    i=0;
    for(const DOMElement* ele=ec->getFirstElementChild(); ele!=0; ele=ele->getNextElementSibling(), i++)
      if(!function)
        m(i)=C(stringToValue(X()%E(ele)->getFirstTextChild()->getData(), ele))->double_value();
      else {
        casadi::SX Mele;
        try { Mele=cast<casadi::SX>(stringToValue(X()%E(ele)->getFirstTextChild()->getData(), ele)); } MBXMLUTILS_RETHROW(e)
        if(Mele.size1()!=1 || Mele.size2()!=1) throw DOMEvalException("Scalar argument required.", e);
        M.elem(i,0)=Mele.elem(0,0);
      }
    if(!function)
      return C(handleUnit(e, make_shared<octave_value>(m)));
    else {
      octave_value octF=createCasADi("SXFunction");
      casadi::SXFunction f(inputs, M);
      try { cast<casadi::SXFunction*>(C(octF))->assignNode(f.get()); } MBXMLUTILS_RETHROW(e)
      return C(octF);
    }
  }
  
  // a XML matrix
  ec=E(e)->getFirstElementChildNamed(PV%"xmlMatrix");
  if(ec) {
    int i, j;
    // calculate nubmer for rows and cols
    i=0;
    for(const DOMElement* row=ec->getFirstElementChild(); row!=0; row=row->getNextElementSibling(), i++);
    j=0;
    for(const DOMElement* ele=ec->getFirstElementChild()->getFirstElementChild(); ele!=0; ele=ele->getNextElementSibling(), j++);
    // get/eval values
    Matrix m(i, j);
    casadi::SX M;
    if(function)
      M.resize(i, j);
    i=0;
    for(const DOMElement* row=ec->getFirstElementChild(); row!=0; row=row->getNextElementSibling(), i++) {
      j=0;
      for(const DOMElement* col=row->getFirstElementChild(); col!=0; col=col->getNextElementSibling(), j++)
        if(!function)
          m(j*m.rows()+i)=C(stringToValue(X()%E(col)->getFirstTextChild()->getData(), col))->double_value();
        else {
          casadi::SX Mele;
          try { Mele=cast<casadi::SX>(stringToValue(X()%E(col)->getFirstTextChild()->getData(), col)); } MBXMLUTILS_RETHROW(e)
          if(Mele.size1()!=1 || Mele.size2()!=1) throw DOMEvalException("Scalar argument required.", e);
          M.elem(i,0)=Mele.elem(0,0);
        }
    }
    if(!function)
      return C(handleUnit(e, make_shared<octave_value>(m)));
    else {
      octave_value octF=createCasADi("SXFunction");
      casadi::SXFunction f(inputs, M);
      try { cast<casadi::SXFunction*>(C(octF))->assignNode(f.get()); } MBXMLUTILS_RETHROW(e)
      return C(octF);
    }
  }
  
  // a element with a single text child (including unit conversion)
  if(!e->getFirstElementChild() &&
     (E(e)->isDerivedFrom(PV%"scalar") ||
      E(e)->isDerivedFrom(PV%"vector") ||
      E(e)->isDerivedFrom(PV%"matrix") ||
      E(e)->isDerivedFrom(PV%"fullOctEval") ||
      function)
    ) {
    shared_ptr<octave_value> ret=C(stringToValue(X()%E(e)->getFirstTextChild()->getData(), e));
    if(E(e)->isDerivedFrom(PV%"scalar") && !ret->is_scalar_type())
      throw DOMEvalException("Octave value is not of type scalar", e);
    if(E(e)->isDerivedFrom(PV%"vector") && ret->columns()!=1)
      throw DOMEvalException("Octave value is not of type vector", e);
    if(E(e)->isDerivedFrom(PV%"stringFullOctEval") && !ret->is_scalar_type() && !ret->is_string()) // also filenameFullOctEval
      throw DOMEvalException("Octave value is not of type scalar string", e);

    // add filenames to dependencies
    if(dependencies && E(e)->isDerivedFrom(PV%"filenameFullOctEval"))
      dependencies->push_back(E(e)->convertPath(ret->string_value()));
  
    // convert unit
    ret=handleUnit(e, ret);
  
    if(!function)
      return ret;
    else {
      octave_value octF=createCasADi("SXFunction");
      try {
        casadi::SXFunction f(inputs, cast<casadi::SX>(ret));
        cast<casadi::SXFunction*>(C(octF))->assignNode(f.get());
      } MBXMLUTILS_RETHROW(e)
      return C(octF);
    }
  }
  
  // rotation about x,y,z
  for(char ch='X'; ch<='Z'; ch++) {
    static octave_function *rotFunc[3]={
      symbol_table::find_function("rotateAboutX").function_value(), // get ones a pointer performance reasons
      symbol_table::find_function("rotateAboutY").function_value(), // get ones a pointer performance reasons
      symbol_table::find_function("rotateAboutZ").function_value()  // get ones a pointer performance reasons
    };
    ec=E(e)->getFirstElementChildNamed(PV%(string("about")+ch));
    if(ec) {
      // convert
      shared_ptr<octave_value> angle=C(eval(ec));
      octave_value_list ret;
      try { ret=fevalThrow(rotFunc[ch-'X'], octave_value_list(*angle), 1,
        string("Unable to generate rotation matrix using rotateAbout")+ch+"."); } MBXMLUTILS_RETHROW(ec)
      return C(ret(0));
    }
  }
  
  // rotation cardan or euler
  for(int i=0; i<2; i++) {
    static const string rotFuncName[2]={
      "cardan",
      "euler"
    };
    static octave_function *rotFunc[2]={
      symbol_table::find_function(rotFuncName[0]).function_value(), // get ones a pointer performance reasons
      symbol_table::find_function(rotFuncName[1]).function_value()  // get ones a pointer performance reasons
    };
    ec=E(e)->getFirstElementChildNamed(PV%rotFuncName[i]);
    if(ec) {
      // convert
      octave_value_list angles;
      const DOMElement *ele;
  
      ele=ec->getFirstElementChild();
      angles.append(*handleUnit(ec, C(eval(ele))));
      ele=ele->getNextElementSibling();
      angles.append(*handleUnit(ec, C(eval(ele))));
      ele=ele->getNextElementSibling();
      angles.append(*handleUnit(ec, C(eval(ele))));
      octave_value_list ret;
      try { ret=fevalThrow(rotFunc[i], angles, 1,
        string("Unable to generate rotation matrix using ")+rotFuncName[i]); } MBXMLUTILS_RETHROW(ec)
      return C(ret(0));
    }
  }
  
  // from file
  ec=E(e)->getFirstElementChildNamed(PV%"fromFile");
  if(ec) {
    static octave_function *loadFunc=symbol_table::find_function("load").function_value();  // get ones a pointer performance reasons
    shared_ptr<octave_value> fileName=C(stringToValue(E(ec)->getAttribute("href"), ec, false));
    if(dependencies)
      dependencies->push_back(E(e)->convertPath(fileName->string_value()));

    // restore current dir on exit and change current dir
    PreserveCurrentDir preserveDir;
    bfs::path chdir=E(e)->getOriginalFilename().parent_path();
    if(!chdir.empty())
      bfs::current_path(chdir);

    octave_value_list ret;
    try { ret=fevalThrow(loadFunc, octave_value_list(*fileName), 1,
      string("Unable to load file ")+E(ec)->getAttribute("href")); } MBXMLUTILS_RETHROW(ec)
    return C(ret(0));
  }
  
  // unknown element: throw
  throw DOMEvalException("Dont know how to evaluate this element", e);
}

shared_ptr<void> OctEval::eval(const xercesc::DOMAttr *a, const xercesc::DOMElement *pe) {
  bool fullEval;
  if(A(a)->isDerivedFrom(PV%"fullOctEval"))
    fullEval=true;
  else if(A(a)->isDerivedFrom(PV%"partialOctEval"))
    fullEval=false;
  else
    throw DOMEvalException("Unknown XML attribute type", pe, a);

  // evaluate attribute fully
  if(fullEval) {
    shared_ptr<octave_value> ret=C(stringToValue(X()%a->getValue(), pe));
    if(A(a)->isDerivedFrom(PV%"floatFullOctEval")) {
      if(!ret->is_scalar_type() || !ret->is_real_type())
        throw DOMEvalException("Value is not of type scalar float", pe, a);
    }
    else if(A(a)->isDerivedFrom(PV%"stringFullOctEval")) {
      if(!ret->is_scalar_type() || !ret->is_string()) // also filenameFullOctEval
        throw DOMEvalException("Value is not of type scalar string", pe, a);
    }
    else if(A(a)->isDerivedFrom(PV%"integerFullOctEval")) {
      bool isInt=true;
      try { boost::lexical_cast<int>(ret->double_value()); } catch(const boost::bad_lexical_cast &) { isInt=false; }
      if(!ret->is_scalar_type() || !ret->is_real_type() || !isInt) // also symbolicFunctionArgDimType
        throw DOMEvalException("Value is not of type scalar integer", pe, a);
    }
    else if(A(a)->isDerivedFrom(PV%"booleanFullOctEval")) {
      if(!ret->is_scalar_type() || !ret->is_real_type() || (ret->double_value()!=0 && ret->double_value()!=1))
        throw DOMEvalException("Value is not of type scalar boolean", pe, a);
    }
    else
      throw DOMEvalException("Unknown XML attribute type for evaluation", pe, a);

    // add filenames to dependencies
    if(dependencies && A(a)->isDerivedFrom(PV%"filenameFullOctEval"))
      dependencies->push_back(E(pe)->convertPath(ret->string_value()));

    return ret;
  }
  // evaluate attribute partially
  else {
    octave_value ret;
    string s=partialStringToOctValue(X()%a->getValue(), pe);
    if(A(a)->isDerivedFrom(PV%"varnamePartialOctEval")) { // also symbolicFunctionArgNameType
      if(s.length()<1)
        throw DOMEvalException("A variable name must consist of at least 1 character", pe, a);
      if(!(s[0]=='_' || ('a'<=s[0] && s[0]<='z') || ('A'<=s[0] && s[0]<='Z')))
        throw DOMEvalException("A variable name start with _, a-z or A-Z", pe, a);
      for(size_t i=1; i<s.length(); i++)
        if(!(s[i]=='_' || ('a'<=s[i] && s[i]<='z') || ('A'<=s[i] && s[i]<='Z')))
          throw DOMEvalException("Only the characters _, a-z, A-Z and 0-9 are allowed for variable names", pe, a);
      ret=s;
    }
    else if(A(a)->isDerivedFrom(PV%"floatPartialOctEval"))
      try { ret=boost::lexical_cast<double>(s); }
      catch(const boost::bad_lexical_cast &) { throw DOMEvalException("Value is not of type scalar float", pe, a); }
    else if(A(a)->isDerivedFrom(PV%"stringPartialOctEval")) // also filenamePartialOctEval
      try { ret=boost::lexical_cast<string>(s); }
      catch(const boost::bad_lexical_cast &) { throw DOMEvalException("Value is not of type scalar string", pe, a); }
    else if(A(a)->isDerivedFrom(PV%"integerPartialOctEval"))
      try { ret=boost::lexical_cast<int>(s); }
      catch(const boost::bad_lexical_cast &) { throw DOMEvalException("Value is not of type scalar integer", pe, a); }
    else if(A(a)->isDerivedFrom(PV%"booleanPartialOctEval"))
      try { ret=boost::lexical_cast<bool>(s); }
      catch(const boost::bad_lexical_cast &) { throw DOMEvalException("Value is not of type scalar boolean", pe, a); }
    else
      throw DOMEvalException("Unknown XML attribute type for evaluation", pe, a);

    // add filenames to dependencies
    if(dependencies && A(a)->isDerivedFrom(PV%"filenamePartialOctEval"))
      dependencies->push_back(E(pe)->convertPath(s));

    return C(ret);
  }
}

OctEval::ValueType OctEval::getType(const shared_ptr<void> &value) {
  if(C(value)->is_scalar_type() && C(value)->is_real_type() && !C(value)->is_string())
    return ScalarType;
  else if(C(value)->is_matrix_type() && C(value)->is_real_type() && !C(value)->is_string()) {
    Matrix m=C(value)->matrix_value();
    if(m.cols()==1)
      return VectorType;
    else
      return MatrixType;
  }
  else {
    // get the casadi type
    static octave_function *swig_type=symbol_table::find_function("swig_type").function_value(); // get ones a pointer to swig_type for performance reasons
    octave_value swigType;
    {
      BLOCK_STDERR(blockstderr);
      swigType=feval(swig_type, *C(value), 1)(0);
    }
    if(error_state!=0) {
      error_state=0;
      if(C(value)->is_string())
        return StringType;
      throw DOMEvalException("The provided value has an unknown type.");
    }
    if(swigType.string_value()=="SX")
      return SXType;
    else if(swigType.string_value()=="DMatrix")
      return DMatrixType;
    else if(swigType.string_value()=="SXFunction")
      return SXFunctionType;
    else
      throw DOMEvalException("The provided value has an unknown type.");
  }
}

octave_value_list OctEval::fevalThrow(octave_function *func, const octave_value_list &arg, int n,
                                       const string &msg) {
  ostringstream err;
  octave_value_list ret;
  {
    REDIR_STDERR(redirstderr, err.rdbuf());
    ret=feval(func, arg, n);
  }
  if(error_state!=0) {
    error_state=0;
    throw DOMEvalException(err.str()+msg);
  }
  return ret;
}

shared_ptr<octave_value> OctEval::handleUnit(const xercesc::DOMElement *e, const shared_ptr<octave_value> &ret) {
  string eqn;
  string unit=E(e)->getAttribute("unit");
  if(!unit.empty())
    eqn=units[unit];
  else {
    string convertUnit=E(e)->getAttribute("convertUnit");
    if(!convertUnit.empty())
      eqn=convertUnit;
    else
      return ret;
  }
  // handle common unit conversions very fast (without octave evaluation)
  if(eqn=="value")
    return ret;
  // all other conversion must be processed using octave
  NewParamLevel newParamLevel(*this, true);
  addParam("value", C(ret));
  return C(stringToValue(eqn, e));
}

// cast octave value to swig object ptr or swig object copy
void* OctEval::castToSwig(const shared_ptr<void> &value) {
  // get the casadi pointer: octave returns a 64bit integer which represents the pointer
  static octave_function *swig_this=symbol_table::find_function("swig_this").function_value(); // get ones a pointer to swig_this for performance reasons
  octave_value swigThis;
  {
    BLOCK_STDERR(blockstderr);
    swigThis=feval(swig_this, *C(value), 1)(0);
  }
  if(error_state!=0)
    throw std::runtime_error("Internal error: Not a swig object");
  void *swigPtr=reinterpret_cast<void*>(swigThis.uint64_scalar_value().value());
  // convert the void pointer to the correct casadi type
  return swigPtr;
}

} // end namespace MBXMLUtils
