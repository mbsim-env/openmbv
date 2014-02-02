#include "mbxmlutils/octeval.h"
#include <stdexcept>
#include <boost/filesystem.hpp>
#include <boost/locale.hpp>
#include <boost/math/special_functions/round.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/regex.hpp>
#include <octave/octave.h>
#include <casadi/symbolic/sx/sx_tools.hpp>
#include <mbxmlutilshelper/dom.h>
#include <mbxmlutilshelper/utils.h>
#include <mbxmlutilshelper/getinstallpath.h>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMAttr.hpp>

//MFMF: should also compile if casadi is not present

using namespace std;
using namespace xercesc;
using namespace boost;
namespace bfs=boost::filesystem;

namespace {
  // NOTE: we can skip the use of utf8Facet (see below) and set the facet globally (for bfs::path and others) using:
  // std::locale::global(boost::locale::generator().generate("UTF8"));
  // boost::filesystem::path::imbue(std::locale());
  const bfs::path::codecvt_type *utf8Facet(&use_facet<bfs::path::codecvt_type>(boost::locale::generator().generate("UTF8")));
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

NewParamLevel::NewParamLevel(OctEval &oe_, bool newLevel_) : oe(oe_), newLevel(newLevel_) {
  if(newLevel)
    oe.pushParams();
}

NewParamLevel::~NewParamLevel() {
  if(newLevel)
    oe.popParams();
}

template<>
string OctEval::cast<string>(const octave_value &value) {
  static const double eps=pow(10, -numeric_limits<double>::digits10-2);
  ValueType type=getType(value);
  ostringstream ret;
  ret.precision(numeric_limits<double>::digits10+1);
  if(type==StringType) {
    ret<<"\""<<value.string_value()<<"\"";
    return ret.str();
  }
  if(type==ScalarType || type==VectorType || type==MatrixType || type==SXMatrixType || type==DMatrixType) {
    vector<vector<double> > m=cast<vector<vector<double> > >(value);
    if(type!=ScalarType)
      ret<<"[";
    for(int i=0; i<m.size(); i++) {
      for(int j=0; j<m[i].size(); j++) {
        long mlong=0;
        try {
           mlong=boost::math::lround(m[i][j]);
        }
        catch(...) {}
        double delta=fabs(mlong-m[i][j]);
        if(delta>eps*m[i][j] && delta>eps)
          ret<<m[i][j]<<(j<m[i].size()-1?",":"");
        else
          ret<<mlong<<(j<m[i].size()-1?",":"");
      }
      ret<<(i<m.size()-1?" ; ":"");
    }
    if(type!=ScalarType || m.size()==0)
      ret<<"]";
    return ret.str();
  }
  throw runtime_error("Can not convert this octave variable to a string.");
}

template<>
long OctEval::cast<long>(const octave_value &value) {
  static const double eps=pow(10, -numeric_limits<double>::digits10-2);
  vector<vector<double> > m=cast<vector<vector<double> > >(value);
  if(getType(value)==ScalarType) {
    long ret=boost::math::lround(m[0][0]);
    double delta=fabs(ret-m[0][0]);
    if(delta>eps*m[0][0] && delta>eps)
      throw runtime_error("Canot cast octave value to long.");
    else
      return ret;
  }
  throw runtime_error("Cannot cast octave value to long.");
}

template<>
double OctEval::cast<double>(const octave_value &value) {
  vector<vector<double> > m=cast<vector<vector<double> > >(value);
  if(getType(value)==ScalarType)
    return m[0][0];
  throw runtime_error("Cannot cast octave value to vector<double>.");
}

template<>
vector<double> OctEval::cast<vector<double> >(const octave_value &value) {
  vector<vector<double> > m=cast<vector<vector<double> > >(value);
  ValueType type=getType(value);
  if(type==ScalarType || type==VectorType) {
    vector<double> ret;
    for(size_t i=0; i<m.size(); i++)
      ret.push_back(m[i][0]);
    return ret;
  }
  throw runtime_error("Cannot cast octave value to vector<double>.");
}

template<>
vector<vector<double> > OctEval::cast<vector<vector<double> > >(const octave_value &value) {
  ValueType type=getType(value);
  if(type==ScalarType || type==VectorType || type==MatrixType) {
    vector<vector<double> > ret;
    Matrix m=value.matrix_value();
    for(int i=0; i<m.rows(); i++) {
      vector<double> row;
      for(int j=0; j<m.cols(); j++)
        row.push_back(m(j*m.rows()+i));
      ret.push_back(row);
    }
    return ret;
  }
  if(type==SXMatrixType || type==DMatrixType) {
    CasADi::SXMatrix m;
    if(type==SXMatrixType)
      m=cast<CasADi::SXMatrix>(value);
    else
      m=cast<CasADi::DMatrix>(value);
    if(!CasADi::isConstant(m))
      throw runtime_error("Can only cast constant value to vector<vector<double> >.");
    vector<vector<double> > ret;
    for(int i=0; i<m.size1(); i++) {
      vector<double> row;
      for(int j=0; j<m.size2(); j++)
        row.push_back(m.elem(i,j).getValue());
      ret.push_back(row);
    }
    return ret;
  }
  throw runtime_error("Cannot cast octave value to vector<vector<double> >.");
}

template<>
DOMElement* OctEval::cast<DOMElement*>(const octave_value &value, DOMDocument *doc) {
  if(getType(value)==SXFunctionType)
    return convertCasADiToXML(cast<CasADi::SXFunction>(value), doc);
  throw runtime_error("Cannot cast octave value to DOMElement*.");
}

template<>
CasADi::SXMatrix OctEval::cast<CasADi::SXMatrix>(const octave_value &value) {
  ValueType type=getType(value);
  if(type==SXMatrixType)
    return castToSwig<CasADi::SXMatrix>(value);
  if(type==DMatrixType)
    return castToSwig<CasADi::DMatrix>(value);
  if(type==ScalarType || type==VectorType || type==MatrixType) {
    vector<vector<double> > m=cast<vector<vector<double> > >(value);
    CasADi::SXMatrix ret(m.size(), m[0].size());
    for(int i=0; i<m.size(); i++)
      for(int j=0; j<m[i].size(); j++)
        ret.elem(i,j)=m[i][j];
    return ret;
  }
  throw runtime_error("Cannot cast to CasADi::SXMatrix*.");
}

template<>
CasADi::SXMatrix* OctEval::cast<CasADi::SXMatrix*>(const octave_value &value) {
  if(getType(value)==SXMatrixType)
    return castToSwig<CasADi::SXMatrix*>(value);
  throw runtime_error("Cannot cast to CasADi::SXMatrix*.");
}

template<>
CasADi::DMatrix OctEval::cast<CasADi::DMatrix>(const octave_value &value) {
  ValueType type=getType(value);
  if(type==DMatrixType)
    return castToSwig<CasADi::DMatrix>(value);
  if(type==ScalarType || type==VectorType || type==MatrixType || type==SXMatrixType) {
    vector<vector<double> > m=cast<vector<vector<double> > >(value);
    CasADi::DMatrix ret(m.size(), m[0].size());
    for(int i=0; i<m.size(); i++)
      for(int j=0; j<m[i].size(); j++)
        ret.elem(i,j)=m[i][j];
    return ret;
  }
  throw runtime_error("Cannot cast to CasADi::DMatrix.");
}

template<>
CasADi::DMatrix* OctEval::cast<CasADi::DMatrix*>(const octave_value &value) {
  if(getType(value)==DMatrixType)
    return castToSwig<CasADi::DMatrix*>(value);
  throw runtime_error("Cannot cast to CasADi::DMatrix*.");
}

template<>
CasADi::SXFunction OctEval::cast<CasADi::SXFunction>(const octave_value &value) {
  if(getType(value)==SXFunctionType)
    return castToSwig<CasADi::SXFunction>(value);
  throw runtime_error("Cannot cast to CasADi::SXFunction.");
}

template<>
CasADi::SXFunction* OctEval::cast<CasADi::SXFunction*>(const octave_value &value) {
  if(getType(value)==SXFunctionType)
    return castToSwig<CasADi::SXFunction*>(value);
  throw runtime_error("Cannot cast to CasADi::SXFunction*.");
}

octave_value OctEval::createCasADi(const string &name) {
  static list<octave_value_list> idx;
  if(idx.empty()) {
    idx.push_back(octave_value_list(name));
    idx.push_back(octave_value_list());
  }
  *idx.begin()=octave_value_list(name);
  return casadiOctValue.subsref(".(", idx);
}

int OctEval::initCount=0;

std::map<std::string, std::string> OctEval::units;

InitXerces OctEval::initXerces;

octave_value OctEval::casadiOctValue;

OctEval::OctEval(vector<bfs::path> *dependencies_) : dependencies(dependencies_) {
  if(initCount==0) {

    bfs::path XMLDIR=MBXMLUtils::getInstallPath()/"share"/"mbxmlutils"/"xml"; // use rel path if build configuration dose not work
  
    static vector<char*> octave_argv;
    octave_argv.resize(6);
    octave_argv[0]=const_cast<char*>("embedded");
    octave_argv[1]=const_cast<char*>("--no-history");
    octave_argv[2]=const_cast<char*>("--no-init-file");
    octave_argv[3]=const_cast<char*>("--no-line-editing");
    octave_argv[4]=const_cast<char*>("--no-window-system");
    octave_argv[5]=const_cast<char*>("--silent");
    octave_main(6, &octave_argv[0], 1);
  
    octave_value_list warnArg;
    warnArg.append("error");
    warnArg.append("Octave:divide-by-zero");
    feval("warning", warnArg);
    if(error_state!=0) { error_state=0; throw runtime_error("Internal error: unable to disable warnings."); }
  
    feval("addpath", octave_value_list(octave_value((MBXMLUtils::getInstallPath()/"share"/"mbxmlutils"/"octave").string(*utf8Facet))));
    if(error_state!=0) { error_state=0; throw runtime_error("Internal error: cannot add octave search path."); }

    feval("addpath", octave_value_list(octave_value((MBXMLUtils::getInstallPath()/"bin").string(*utf8Facet))));
    if(error_state!=0) { error_state=0; throw string("Internal error: cannot add casadi octave search path."); }

    {
      BLOCK_STDERR(blockstderr);
      casadiOctValue=feval("swigLocalLoad", octave_value_list("casadi"), 1)(0);
      if(error_state!=0) { error_state=0; throw string("Internal error: unable to initialize casadi."); }
    }

    // get units
    cout<<"Build unit list for measurements."<<endl;
    boost::shared_ptr<DOMDocument> mmdoc=DOMParser::create(false)->parse(XMLDIR/"measurement.xml");
    DOMElement *ele, *el2;
    for(ele=mmdoc->getDocumentElement()->getFirstElementChild(); ele!=0; ele=ele->getNextElementSibling())
      for(el2=ele->getFirstElementChild(); el2!=0; el2=el2->getNextElementSibling()) {
        if(units.find(E(el2)->getAttribute("name"))!=units.end())
          throw runtime_error(string("Internal error: Unit name ")+E(el2)->getAttribute("name")+" is defined more than once.");
        units[E(el2)->getAttribute("name")]=X()%E(el2)->getFirstTextChild()->getData();
      }
  }
  initCount++;
};

OctEval::~OctEval() {
  initCount--;
  if(initCount==0) {
    //Workaround: eval a VALID dummy statement before leaving "main" to prevent a crash in post main
    int dummy;
    eval_string("1+1", true, dummy, 0); // eval as statement list
  }
}

void OctEval::addParam(const std::string &paramName, const octave_value& value) {
  currentParam[paramName]=value;
}

void OctEval::addParamSet(const DOMElement *e) {
  // outer loop to resolve recursive parameters
  list<const DOMElement*> c;
  for(const DOMElement *ee=e->getFirstElementChild(); ee!=NULL; ee=ee->getNextElementSibling())
    c.push_back(ee);
  size_t length=c.size();
  for(size_t outerLoop=0; outerLoop<length; outerLoop++) {
    // evaluate parameter
    list<const DOMElement*>::iterator ee=c.begin();
    while(ee!=c.end()) {
      int err=0;
      octave_value ret;
      try { 
        BLOCK_STDERR(blockstderr);
        ret=eval(*ee);
      }
      catch(const std::exception &ex) {
        err=1;
      }
      if(err==0) { // if no error
        addParam(E(*ee)->getAttribute("name"), ret); // add param to list
        list<const DOMElement*>::iterator eee=ee; eee++;
        c.erase(ee);
        ee=eee;
      }
      else
        ee++;
    }
  }
  if(c.size()>0) { // if parameters are left => error
    DOMEvalExceptionList error;
    error.push_back(DOMEvalException("Error in one of the following parameters or infinit loop in this parameters:"));
    for(list<const DOMElement*>::iterator ee=c.begin(); ee!=c.end(); ee++) {
      try {
        eval(*ee);
      }
      catch(const DOMEvalException &ex) {
        error.push_back(ex);
      }
      catch(const std::exception &ex) {
        error.push_back(DOMEvalException(ex.what()));
      }
    }
    error.push_back(DOMEvalException("Error processing parameters. See above."));
    throw error;
  }
}

void OctEval::pushParams() {
  paramStack.push(currentParam);
}

void OctEval::popParams() {
  currentParam=paramStack.top();
  paramStack.pop();
}

void OctEval::addPath(const bfs::path &dir) {
  static octave_function *addpath=symbol_table::find_function("addpath").function_value();  // get ones a pointer performance reasons
  fevalThrow(addpath, octave_value_list(octave_value(dir.generic_string())));
}

octave_value OctEval::stringToOctValue(const string &str, const DOMElement *e, bool fullEval) const {
  if(fullEval)
    return fullStringToOctValue(str, e);
  else
    return partialStringToOctValue(str, e);
}

octave_value OctEval::fullStringToOctValue(const string &str, const DOMElement *e) const {
  // restore current dir on exit and change current dir
  PreserveCurrentDir preserveDir;
  if(e) {
    bfs::path chdir=E(e)->getOriginalFilename().parent_path();
    if(!chdir.empty())
      bfs::current_path(chdir);
  }

  // clear octave
  symbol_table::clear_variables();
  // restore current parameters
  for(map<string, octave_value>::const_iterator i=currentParam.begin(); i!=currentParam.end(); i++)
    symbol_table::varref(i->first)=i->second;

  ostringstream err;
  try{
    int dummy;
    BLOCK_STDOUT(blockstdout);
    REDIR_STDERR(redirstderr, err.rdbuf());
    eval_string(str, true, dummy, 0); // eval as statement list
  }
  catch(const std::exception &ex) {
    error_state=0;
    throw DOMEvalException(err.str()+ex.what(), e);
  }
  catch(...) {
    error_state=0;
    throw;
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
    ret=symbol_table::varref(strNoSpace);
  else if(!symbol_table::is_variable("ret"))
    ret=symbol_table::varref("ans");
  else
    ret=symbol_table::varref("ret");

  return ret;
}

string OctEval::partialStringToOctValue(const string &str, const DOMElement *e) const {
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
    string subst=cast<string>(ret);
    if(getType(ret)==StringType)
      subst=subst.substr(1, subst.length()-2);
    s=s.substr(0,i)+subst+s.substr(j+1);
  }
  return s;
}

octave_value OctEval::eval(const xercesc::DOMElement *e) {
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
  vector<CasADi::SXMatrix> inputs;
  if(function) {
    addParam("casadi", casadiOctValue);
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

      octave_value octArg=createCasADi("SXMatrix");
      CasADi::SXMatrix *arg=cast<CasADi::SXMatrix*>(octArg);
      *arg=CasADi::ssym(X()%a->getValue(), dim, 1);
      addParam(X()%a->getValue(), octArg);
      inputs.resize(max(nr, static_cast<int>(inputs.size()))); // fill new elements with default ctor (isNull()==true)
      inputs[nr-1]=*arg;
    }
    // check if one argument was not set. If so error
    for(int i=0; i<inputs.size(); i++)
      if(inputs[i].isNull()) // a isNull() object is a error (see above), since not all arg?name args were defined
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
    CasADi::SXMatrix M;
    if(function)
      M.resize(i, 1);
    i=0;
    for(const DOMElement* ele=ec->getFirstElementChild(); ele!=0; ele=ele->getNextElementSibling(), i++)
      if(!function)
        m(i)=stringToOctValue(X()%E(ele)->getFirstTextChild()->getData(), ele).double_value();
      else {
        CasADi::SXMatrix Mele=cast<CasADi::SXMatrix>(stringToOctValue(X()%E(ele)->getFirstTextChild()->getData(), ele));
        if(Mele.size1()!=1 || Mele.size2()!=1) throw DOMEvalException("Scalar argument required.", e);
        M.elem(i,0)=Mele.elem(0,0);
      }
    if(!function)
      return handleUnit(e, m);
    else {
      octave_value octF=createCasADi("SXFunction");
      CasADi::SXFunction f(inputs, M);
      cast<CasADi::SXFunction*>(octF)->assignNode(f.get());
      return octF;
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
    CasADi::SXMatrix M;
    if(function)
      M.resize(i, j);
    i=0;
    for(const DOMElement* row=ec->getFirstElementChild(); row!=0; row=row->getNextElementSibling(), i++) {
      j=0;
      for(const DOMElement* col=row->getFirstElementChild(); col!=0; col=col->getNextElementSibling(), j++)
        if(!function)
          m(j*m.rows()+i)=stringToOctValue(X()%E(col)->getFirstTextChild()->getData(), col).double_value();
        else {
          CasADi::SXMatrix Mele=cast<CasADi::SXMatrix>(stringToOctValue(X()%E(col)->getFirstTextChild()->getData(), col));
          if(Mele.size1()!=1 || Mele.size2()!=1) throw DOMEvalException("Scalar argument required.", e);
          M.elem(i,0)=Mele.elem(0,0);
        }
    }
    if(!function)
      return handleUnit(e, m);
    else {
      octave_value octF=createCasADi("SXFunction");
      CasADi::SXFunction f(inputs, M);
      cast<CasADi::SXFunction*>(octF)->assignNode(f.get());
      return octF;
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
    octave_value ret=stringToOctValue(X()%E(e)->getFirstTextChild()->getData(), e);
    if(E(e)->isDerivedFrom(PV%"scalar") && !ret.is_scalar_type())
      throw DOMEvalException("Octave value is not of type scalar", e);
    if(E(e)->isDerivedFrom(PV%"vector") && ret.columns()!=1)
      throw DOMEvalException("Octave value is not of type vector", e);
    if(E(e)->isDerivedFrom(PV%"stringFullOctEval") && !ret.is_scalar_type() && !ret.is_string())
      throw DOMEvalException("Octave value is not of type scalar string", e);
  
    // convert unit
    ret=handleUnit(e, ret);
  
    if(!function)
      return ret;
    else {
      octave_value octF=createCasADi("SXFunction");
      CasADi::SXFunction f(inputs, cast<CasADi::SXMatrix>(ret));
      cast<CasADi::SXFunction*>(octF)->assignNode(f.get());
      return octF;
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
      octave_value angle=eval(ec);
      octave_value_list ret=fevalThrow(rotFunc[ch-'X'], octave_value_list(angle), 1, string("Unable to generate rotation matrix using rotateAbout")+ch+".", e);
      return ret(0);
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
      angles.append(handleUnit(ec, eval(ele)));
      ele=ele->getNextElementSibling();
      angles.append(handleUnit(ec, eval(ele)));
      ele=ele->getNextElementSibling();
      angles.append(handleUnit(ec, eval(ele)));
      octave_value_list ret=fevalThrow(rotFunc[i], angles, 1, string("Unable to generate rotation matrix using ")+rotFuncName[i], e);
      return ret(0);
    }
  }
  
  // from file
  ec=E(e)->getFirstElementChildNamed(PV%"fromFile");
  if(ec) {
    static octave_function *loadFunc=symbol_table::find_function("load").function_value();  // get ones a pointer performance reasons
    octave_value fileName=stringToOctValue(E(ec)->getAttribute("href"), ec);
    if(dependencies)
      dependencies->push_back(bfs::absolute(fileName.string_value(), E(e)->getOriginalFilename().parent_path()));

    // restore current dir on exit and change current dir
    PreserveCurrentDir preserveDir;
    bfs::path chdir=E(e)->getOriginalFilename().parent_path();
    if(!chdir.empty())
      bfs::current_path(chdir);

    octave_value_list ret=fevalThrow(loadFunc, octave_value_list(fileName), 1, string("Unable to load file ")+E(ec)->getAttribute("href"), e);
    return ret(0);
  }
  
  // unknown element: throw
  throw DOMEvalException("Dont know how to evaluate this element", e);
}

octave_value OctEval::eval(const xercesc::DOMAttr *a, const xercesc::DOMElement *pe) {
  bool fullEval;
  if(A(a)->isDerivedFrom(PV%"fullOctEval"))
    fullEval=true;
  else if(A(a)->isDerivedFrom(PV%"partialOctEval"))
    fullEval=false;
  else
    throw DOMEvalException("Unknown XML attribute type for evaluation", pe);

  // evaluate attribute fully
  if(fullEval) {
    octave_value ret=stringToOctValue(X()%a->getValue(), pe);
    if(A(a)->isDerivedFrom(PV%"floatFullOctEval") && (!ret.is_scalar_type() || !ret.is_double_type()))
      throw DOMEvalException("Octave value is not of type scalar float", pe);
    if(A(a)->isDerivedFrom(PV%"stringFullOctEval") && (!ret.is_scalar_type() && !ret.is_string()))
      throw DOMEvalException("Octave value is not of type scalar string", pe);
    if(A(a)->isDerivedFrom(PV%"integerFullOctEval") && (!ret.is_scalar_type() && !ret.is_integer_type())) // also symbolicFunctionArgDimType
      throw DOMEvalException("Octave value is not of type scalar integer", pe);
    if(A(a)->isDerivedFrom(PV%"booleanFullOctEval") && (!ret.is_scalar_type() && !ret.is_bool_scalar()))
      throw DOMEvalException("Octave value is not of type scalar boolean", pe);
    return ret;
  }
  // evaluate attribute partially
  else {
    string s=partialStringToOctValue(X()%a->getValue(), pe);
    if(A(a)->isDerivedFrom(PV%"varnamePartialOctEval")) // also symbolicFunctionArgNameType
      try {
        string str=boost::lexical_cast<string>(s);
        static regex varnameRegex("[_a-zA-Z][_a-zA-Z0-9]*");
        if(!regex_match(str, varnameRegex))
          throw DOMEvalException("Octave value is not of type variable name ([a-zA-Z][a-zA-Z0-9_]*)", pe);
      }
      catch(const boost::bad_lexical_cast &) { throw DOMEvalException("Octave value is not of type scalar string", pe); }
    if(A(a)->isDerivedFrom(PV%"floatPartialOctEval"))
      try { return boost::lexical_cast<double>(s); }
      catch(const boost::bad_lexical_cast &) { throw DOMEvalException("Octave value is not of type scalar float", pe); }
    if(A(a)->isDerivedFrom(PV%"stringPartialOctEval")) // also filenamePartialOctEval
      try { return boost::lexical_cast<string>(s); }
      catch(const boost::bad_lexical_cast &) { throw DOMEvalException("Octave value is not of type scalar string", pe); }
    if(A(a)->isDerivedFrom(PV%"integerPartialOctEval"))
      try { return boost::lexical_cast<int>(s); }
      catch(const boost::bad_lexical_cast &) { throw DOMEvalException("Octave value is not of type scalar integer", pe); }
    if(A(a)->isDerivedFrom(PV%"booleanPartialOctEval"))
      try { return boost::lexical_cast<bool>(s); }
      catch(const boost::bad_lexical_cast &) { throw DOMEvalException("Octave value is not of type scalar boolean", pe); }
    throw DOMEvalException("Unknown XML attribute type for evaluation", pe);
  }
}

OctEval::ValueType OctEval::getType(const octave_value &value) {
  if(value.is_scalar_type() && value.is_real_type() && !value.is_string())
    return ScalarType;
  else if(value.is_matrix_type() && value.is_real_type() && !value.is_string()) {
    Matrix m=value.matrix_value();
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
      swigType=feval(swig_type, value, 1)(0);
    }
    if(error_state!=0) {
      error_state=0;
      if(value.is_string())
        return StringType;
      throw runtime_error("The provided octave value has an unknown type.");
    }
    if(swigType.string_value()=="SXMatrix")
      return SXMatrixType;
    else if(swigType.string_value()=="DMatrix")
      return DMatrixType;
    else if(swigType.string_value()=="SXFunction")
      return SXFunctionType;
    else
      throw runtime_error("The provided octave value has an unknown type.");
  }
}

octave_value_list OctEval::fevalThrow(octave_function *func, const octave_value_list &arg, int n,
                                       const string &msg, const DOMElement *e) {
  ostringstream err;
  octave_value_list ret;
  {
    REDIR_STDERR(redirstderr, err.rdbuf());
    ret=feval(func, arg, n);
  }
  if(error_state!=0) {
    error_state=0;
    if(!e)
      throw runtime_error(err.str()+msg);
    else
      throw DOMEvalException(err.str()+msg, e);
  }
  return ret;
}

octave_value OctEval::handleUnit(const xercesc::DOMElement *e, const octave_value &ret) {
  if(!E(e)->getAttribute("unit").empty() || !E(e)->getAttribute("convertUnit").empty()) {
    OctEval oe;
    oe.addParam("value", ret);
    if(!E(e)->getAttribute("unit").empty()) // convert with predefined unit
      return oe.stringToOctValue(units[E(e)->getAttribute("unit")], e);
    if(!E(e)->getAttribute("convertUnit").empty()) // convert with user defined unit
      return oe.stringToOctValue(E(e)->getAttribute("convertUnit"), e);
  }
  return ret;
}

} // end namespace MBXMLUtils
