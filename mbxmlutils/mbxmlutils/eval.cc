#include <config.h>
#include "mbxmlutils/eval.h"
#include "mbxmlutils/octeval.h"
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMAttr.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <mbxmlutilshelper/dom.h>
#include <mbxmlutilshelper/getinstallpath.h>
#include <boost/lexical_cast.hpp>

using namespace std;
using namespace casadi;
using namespace boost;
using namespace xercesc;
namespace bfs=boost::filesystem;

namespace MBXMLUtils {

bool deactivateBlock=getenv("MBXMLUTILS_DEACTIVATE_BLOCK")!=NULL;

NewParamLevel::NewParamLevel(Eval &oe_, bool newLevel_) : oe(oe_), newLevel(newLevel_) {
  if(newLevel) {
    oe.pushParams();
    oe.pushPath();
  }
}

NewParamLevel::~NewParamLevel() {
  if(newLevel) {
    oe.popParams();
    oe.popPath();
  }
}

shared_ptr<void> Eval::casadiValue;

std::map<std::string, std::string> Eval::units;

Eval::Eval(vector<bfs::path> *dependencies_) : dependencies(dependencies_) {
  static bool initialized=false;
  if(!initialized) {
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
    initialized=true;
  }
};

boost::shared_ptr<Eval> Eval::createEvaluator(const string &evalName, vector<bfs::path> *dependencies_) {
  if(evalName=="octave")
    return shared_ptr<OctEval>(new OctEval(dependencies_));
  throw runtime_error("Unknown evaluator "+evalName);
}

Eval::~Eval() {
}

template<>
string Eval::cast<string>(const shared_ptr<void> &value) {
  return cast_string(value);
}

template<>
int Eval::cast<int>(const shared_ptr<void> &value) {
  return cast_int(value);
}

template<>
double Eval::cast<double>(const shared_ptr<void> &value) {
  return cast_double(value);
}

template<>
vector<double> Eval::cast<vector<double> >(const shared_ptr<void> &value) {
  return cast_vector_double(value);
}

template<>
vector<vector<double> > Eval::cast<vector<vector<double> > >(const shared_ptr<void> &value) {
  return cast_vector_vector_double(value);
}

template<>
SX Eval::cast<SX>(const shared_ptr<void> &value) {
  return cast_SX(value);
}

template<>
SX* Eval::cast<SX*>(const shared_ptr<void> &value) {
  return cast_SX_p(value);
}

template<>
SXFunction Eval::cast<SXFunction>(const shared_ptr<void> &value) {
  return cast_SXFunction(value);
}

template<>
SXFunction* Eval::cast<SXFunction*>(const shared_ptr<void> &value) {
  return cast_SXFunction_p(value);
}

template<>
DMatrix Eval::cast<DMatrix>(const shared_ptr<void> &value) {
  return cast_DMatrix(value);
}

template<>
DMatrix* Eval::cast<DMatrix*>(const shared_ptr<void> &value) {
  return cast_DMatrix_p(value);
}

template<>
DOMElement* Eval::cast<DOMElement*>(const shared_ptr<void> &value, DOMDocument *doc) {
  return cast_DOMElement_p(value, doc);
}

void Eval::pushParams() {
  paramStack.push(currentParam);
}

void Eval::popParams() {
  currentParam=paramStack.top();
  paramStack.pop();
}

void Eval::pushPath() {
  pathStack.push(pathStack.top());
}

void Eval::popPath() {
  pathStack.pop();
}

template<>
shared_ptr<void> Eval::create<double>(const double& v) {
  return create_double(v);
}

template<>
shared_ptr<void> Eval::create<vector<double> >(const vector<double>& v) {
  return create_vector_double(v);
}

template<>
shared_ptr<void> Eval::create<vector<vector<double> > >(const vector<vector<double> >& v) {
  return create_vector_vector_double(v);
}

template<>
shared_ptr<void> Eval::create<string>(const string& v) {
  return create_string(v);
}

void Eval::addParam(const std::string &paramName, const shared_ptr<void>& value) {
  currentParam[paramName]=value;
}

void Eval::addParamSet(const DOMElement *e) {
  for(const DOMElement *ee=e->getFirstElementChild(); ee!=NULL; ee=ee->getNextElementSibling()) {
    if(E(ee)->getTagName()==PV%"searchPath") {
      shared_ptr<void> ret=eval(E(ee)->getAttributeNode("href"), ee);
      if(getType(ret)!=StringType)
        throw DOMEvalException("Expecting a variable of type string.", e);
      string str=cast<string>(ret);
      try { addPath(E(ee)->convertPath(str.substr(1, str.length()-2)), ee); } MBXMLUTILS_RETHROW(e)
    }
    else {
      shared_ptr<octave_value> ret=C(eval(ee));
      addParam(E(ee)->getAttribute("name"), ret);
    }
  }
}

shared_ptr<void> Eval::eval(const DOMElement *e) {
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
    addParam("casadi", casadiValue);
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

      shared_ptr<void> casadiSX=createCasADi("SX");
      casadi::SX *arg;
      try { arg=cast<casadi::SX*>(casadiSX); } MBXMLUTILS_RETHROW(e)
      *arg=casadi::SX::sym(X()%a->getValue(), dim, 1);
      addParam(X()%a->getValue(), casadiSX);
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
    vector<double> m(i);
    casadi::SX M;
    if(function)
      M.resize(i, 1);
    i=0;
    for(const DOMElement* ele=ec->getFirstElementChild(); ele!=0; ele=ele->getNextElementSibling(), i++)
      if(!function)
        m[i]=cast<double>(stringToValue(X()%E(ele)->getFirstTextChild()->getData(), ele));
      else {
        casadi::SX Mele;
        try { Mele=cast<casadi::SX>(stringToValue(X()%E(ele)->getFirstTextChild()->getData(), ele)); } MBXMLUTILS_RETHROW(e)
        if(Mele.size1()!=1 || Mele.size2()!=1) throw DOMEvalException("Scalar argument required.", e);
        M.elem(i,0)=Mele.elem(0,0);
      }
    if(!function)
      return handleUnit(e, create(m));
    else {
      shared_ptr<void> func=createCasADi("SXFunction");
      casadi::SXFunction f(inputs, M);
      try { cast<casadi::SXFunction*>(func)->assignNode(f.get()); } MBXMLUTILS_RETHROW(e)
      return func;
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
    vector<vector<double> > m(i, vector<double>(j));
    casadi::SX M;
    if(function)
      M.resize(i, j);
    i=0;
    for(const DOMElement* row=ec->getFirstElementChild(); row!=0; row=row->getNextElementSibling(), i++) {
      j=0;
      for(const DOMElement* col=row->getFirstElementChild(); col!=0; col=col->getNextElementSibling(), j++)
        if(!function)
          m[i][j]=cast<double>(stringToValue(X()%E(col)->getFirstTextChild()->getData(), col));
        else {
          casadi::SX Mele;
          try { Mele=cast<casadi::SX>(stringToValue(X()%E(col)->getFirstTextChild()->getData(), col)); } MBXMLUTILS_RETHROW(e)
          if(Mele.size1()!=1 || Mele.size2()!=1) throw DOMEvalException("Scalar argument required.", e);
          M.elem(i,0)=Mele.elem(0,0);
        }
    }
    if(!function)
      return handleUnit(e, create(m));
    else {
      shared_ptr<void> func=createCasADi("SXFunction");
      casadi::SXFunction f(inputs, M);
      try { cast<casadi::SXFunction*>(func)->assignNode(f.get()); } MBXMLUTILS_RETHROW(e)
      return func;
    }
  }
  
  // a element with a single text child (including unit conversion)
  if(!e->getFirstElementChild() &&
     (E(e)->isDerivedFrom(PV%"scalar") ||
      E(e)->isDerivedFrom(PV%"vector") ||
      E(e)->isDerivedFrom(PV%"matrix") ||
      E(e)->isDerivedFrom(PV%"fullEval") ||
      function)
    ) {
    shared_ptr<void> ret=stringToValue(X()%E(e)->getFirstTextChild()->getData(), e);
    ValueType retType=getType(ret);
    if(E(e)->isDerivedFrom(PV%"scalar") && retType!=ScalarType)
      throw DOMEvalException("Octave value is not of type scalar", e);
    if(E(e)->isDerivedFrom(PV%"vector") && retType!=VectorType && retType!=ScalarType)
      throw DOMEvalException("Octave value is not of type vector", e);
    if(E(e)->isDerivedFrom(PV%"matrix") && retType!=MatrixType && retType!=VectorType && retType!=ScalarType)
      throw DOMEvalException("Octave value is not of type matrix", e);
    if(E(e)->isDerivedFrom(PV%"stringFullEval") && retType!=StringType) // also filenameFullEval
      throw DOMEvalException("Octave value is not of type scalar string", e);

    // add filenames to dependencies
    if(dependencies && E(e)->isDerivedFrom(PV%"filenameFullEval")) {
      if(getType(ret)!=StringType)
        throw DOMEvalException("Value must be of type string", e);
      string str=cast<string>(ret);
      dependencies->push_back(E(e)->convertPath(str.substr(1, str.length()-2)));
    }
  
    // convert unit
    ret=handleUnit(e, ret);
  
    if(!function)
      return ret;
    else {
      shared_ptr<void> func=createCasADi("SXFunction");
      try {
        casadi::SXFunction f(inputs, cast<casadi::SX>(ret));
        cast<casadi::SXFunction*>(func)->assignNode(f.get());
      } MBXMLUTILS_RETHROW(e)
      return func;
    }
  }
  
  // rotation about x,y,z
  for(char ch='X'; ch<='Z'; ch++) {
    ec=E(e)->getFirstElementChildNamed(PV%(string("about")+ch));
    if(ec) {
      // convert
      shared_ptr<void> angle=eval(ec);
      vector<shared_ptr<void> > args(1);
      args[0]=angle;
      shared_ptr<void> ret;
      try { ret=callFunction(string("rotateAbout")+ch, args); } MBXMLUTILS_RETHROW(ec)
      return ret;
    }
  }
  
  // rotation cardan or euler
  for(int i=0; i<2; i++) {
    static const string rotFuncName[2]={
      "cardan",
      "euler"
    };
    ec=E(e)->getFirstElementChildNamed(PV%rotFuncName[i]);
    if(ec) {
      // convert
      vector<shared_ptr<void> > angles(3);
      const DOMElement *ele;
  
      ele=ec->getFirstElementChild();
      angles[0]=handleUnit(ec, eval(ele));
      ele=ele->getNextElementSibling();
      angles[1]=handleUnit(ec, eval(ele));
      ele=ele->getNextElementSibling();
      angles[2]=handleUnit(ec, eval(ele));
      shared_ptr<void> ret;
      try { ret=callFunction(rotFuncName[i], angles); } MBXMLUTILS_RETHROW(ec)
      return ret;
    }
  }
  
  // from file
  ec=E(e)->getFirstElementChildNamed(PV%"fromFile");
  if(ec) {
    shared_ptr<void> fileName=stringToValue(E(ec)->getAttribute("href"), ec, false);
    if(getType(fileName)!=StringType)
      throw DOMEvalException("Value must be of type string", ec);
    string fileNameStr=cast<string>(fileName);
    fileNameStr=fileNameStr.substr(1, fileNameStr.length()-2);
    if(dependencies)
      dependencies->push_back(E(e)->convertPath(fileNameStr));

    // restore current dir on exit and change current dir
    PreserveCurrentDir preserveDir;
    bfs::path chdir=E(e)->getOriginalFilename().parent_path();
    if(!chdir.empty())
      bfs::current_path(chdir);

    shared_ptr<void> ret;
    vector<shared_ptr<void> > args(1);
    args[0]=fileName;
    try { ret=callFunction("load", args); } MBXMLUTILS_RETHROW(ec)
    return ret;
  }
  
  // unknown element: throw
  throw DOMEvalException("Dont know how to evaluate this element", e);
}

shared_ptr<void> Eval::eval(const xercesc::DOMAttr *a, const xercesc::DOMElement *pe) {
  bool fullEval;
  if(A(a)->isDerivedFrom(PV%"fullEval"))
    fullEval=true;
  else if(A(a)->isDerivedFrom(PV%"partialEval"))
    fullEval=false;
  else
    throw DOMEvalException("Unknown XML attribute type", pe, a);

  // evaluate attribute fully
  if(fullEval) {
    shared_ptr<void> ret=stringToValue(X()%a->getValue(), pe);
    ValueType retType=getType(ret);
    if(A(a)->isDerivedFrom(PV%"floatFullEval")) {
      if(retType!=ScalarType)
        throw DOMEvalException("Value is not of type scalar float", pe, a);
    }
    else if(A(a)->isDerivedFrom(PV%"stringFullEval")) {
      if(retType!=StringType) // also filenameFullEval
        throw DOMEvalException("Value is not of type scalar string", pe, a);
    }
    else if(A(a)->isDerivedFrom(PV%"integerFullEval")) {
      try { cast<int>(ret); } MBXMLUTILS_RETHROW(pe);
    }
    else if(A(a)->isDerivedFrom(PV%"booleanFullEval")) {
      int value;
      try { value=cast<int>(ret); } MBXMLUTILS_RETHROW(pe);
      if(value!=0 && value!=1)
        throw DOMEvalException("Value is not of type scalar boolean", pe, a);
    }
    else
      throw DOMEvalException("Unknown XML attribute type for evaluation", pe, a);

    // add filenames to dependencies
    if(dependencies && A(a)->isDerivedFrom(PV%"filenameFullEval")) {
      string str=cast<string>(ret);
      dependencies->push_back(E(pe)->convertPath(str.substr(1, str.length()-2)));
    }

    return ret;
  }
  // evaluate attribute partially
  else {
    shared_ptr<void> ret;
    string s=partialStringToString(X()%a->getValue(), pe);
    if(A(a)->isDerivedFrom(PV%"varnamePartialEval")) { // also symbolicFunctionArgNameType
      if(s.length()<1)
        throw DOMEvalException("A variable name must consist of at least 1 character", pe, a);
      if(!(s[0]=='_' || ('a'<=s[0] && s[0]<='z') || ('A'<=s[0] && s[0]<='Z')))
        throw DOMEvalException("A variable name start with _, a-z or A-Z", pe, a);
      for(size_t i=1; i<s.length(); i++)
        if(!(s[i]=='_' || ('a'<=s[i] && s[i]<='z') || ('A'<=s[i] && s[i]<='Z')))
          throw DOMEvalException("Only the characters _, a-z, A-Z and 0-9 are allowed for variable names", pe, a);
      ret=create(s);
    }
    else if(A(a)->isDerivedFrom(PV%"floatPartialEval"))
      try { ret=create(boost::lexical_cast<double>(s)); }
      catch(const boost::bad_lexical_cast &) { throw DOMEvalException("Value is not of type scalar float", pe, a); }
    else if(A(a)->isDerivedFrom(PV%"stringPartialEval")) // also filenamePartialEval
      try { ret=create(s); }
      catch(const boost::bad_lexical_cast &) { throw DOMEvalException("Value is not of type scalar string", pe, a); }
    else if(A(a)->isDerivedFrom(PV%"integerPartialEval"))
      try { ret=create<double>(boost::lexical_cast<int>(s)); }
      catch(const boost::bad_lexical_cast &) { throw DOMEvalException("Value is not of type scalar integer", pe, a); }
    else if(A(a)->isDerivedFrom(PV%"booleanPartialEval"))
      try { ret=create<double>(boost::lexical_cast<bool>(s)); }
      catch(const boost::bad_lexical_cast &) { throw DOMEvalException("Value is not of type scalar boolean", pe, a); }
    else
      throw DOMEvalException("Unknown XML attribute type for evaluation", pe, a);

    // add filenames to dependencies
    if(dependencies && A(a)->isDerivedFrom(PV%"filenamePartialEval"))
      dependencies->push_back(E(pe)->convertPath(s));

    return ret;
  }
}

shared_ptr<void> Eval::handleUnit(const xercesc::DOMElement *e, const shared_ptr<void> &ret) {
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

string Eval::partialStringToString(const string &str, const DOMElement *e) {
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
    
    shared_ptr<void> ret=fullStringToValue(evalStr, e);
    string subst;
    try {
      subst=cast<string>(ret);
      if(getType(ret)==StringType)
        subst=subst.substr(1, subst.length()-2);
    } MBXMLUTILS_RETHROW(e)
    s=s.substr(0,i)+subst+s.substr(j+1);
  }
  return s;
}

shared_ptr<void> Eval::stringToValue(const string &str, const DOMElement *e, bool fullEval) {
  if(fullEval)
    return fullStringToValue(str, e);
  else
    return create(partialStringToString(str, e));
}

} // end namespace MBXMLUtils
