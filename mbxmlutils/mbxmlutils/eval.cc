#include <config.h>
#include "mbxmlutils/eval.h"
#include "mbxmlutils/eval_static.h"
#include <utility>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMAttr.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <fmatvec/toString.h>
#include <mbxmlutilshelper/utils.h>
#include "mbxmlutilshelper/shared_library.h"
#include <boost/phoenix/object/construct.hpp>
#include <boost/spirit/include/qi_symbols.hpp>
#include <boost/spirit/include/qi_rule.hpp>
#include <boost/spirit/include/qi_char_.hpp>
#include <boost/spirit/include/qi.hpp>
#include <boost/spirit/include/phoenix_operator.hpp>
#include <boost/algorithm/string/trim.hpp>

using namespace std;
using namespace xercesc;
using namespace MBXMLUtils;
namespace bfs=boost::filesystem;

namespace {
#ifndef _WIN32
  const bfs::path LIBBIN("lib");
  const string SHEXT=".so";
#else
  const bfs::path LIBBIN("bin");
  const string SHEXT=".dll";
#endif

class ValueUserDataHandler : public xercesc::DOMUserDataHandler {
  public:
    void handle(DOMOperationType operation, const XMLCh* key, void *data, const xercesc::DOMNode *src, xercesc::DOMNode *dst) override;
};

ValueUserDataHandler valueUserDataHandler;
const string evalValueKey("http://www.mbsim-env.de/dom/MBXMLUtils/evalValue");

void ValueUserDataHandler::handle(DOMUserDataHandler::DOMOperationType operation, const XMLCh* const key,
  void *data, const DOMNode *src, DOMNode *dst) {
  if(MBXMLUtils::X()%key==evalValueKey) {
    if(operation==NODE_DELETED) {
      delete static_cast<MBXMLUtils::Eval::Value*>(data);
      return;
    }
    if((operation==NODE_CLONED || operation==NODE_IMPORTED) &&
       src->getNodeType()==DOMNode::ELEMENT_NODE && dst->getNodeType()==DOMNode::ELEMENT_NODE) {
      dst->setUserData(MBXMLUtils::X()%evalValueKey,
        new MBXMLUtils::Eval::Value(*static_cast<MBXMLUtils::Eval::Value*>(data)), &valueUserDataHandler);
      return;
    }
  }
  throw runtime_error("Internal error: Unknown user data handling: op="+fmatvec::toString(operation)+", key="+MBXMLUtils::X()%key+
                      ", src="+fmatvec::toString(src!=nullptr)+", dst="+fmatvec::toString(dst!=nullptr)+
                      (src ? ", srcType="+fmatvec::toString(src->getNodeType()) : "")+
                      (dst ? ", dstType="+fmatvec::toString(dst->getNodeType()) : ""));
}

ThisLineLocation loc;

}

vector<string> mbxmlutilsStaticDependencies;
boost::filesystem::path originalFilename;

namespace MBXMLUtils {

boost::filesystem::path Eval::installPath(boost::filesystem::path(loc()).parent_path().parent_path());

bool tryDouble2Int(double d, int &i) {
  static const double eps=pow(10, -numeric_limits<double>::digits10-2);

  i=lround(d);
  double delta=fabs(d-i);
  if(delta>eps*i && delta>eps)
    return false;
  return true;
}

NewParamLevel::NewParamLevel(shared_ptr<Eval> oe_, bool newLevel_) : oe(move(oe_)), newLevel(newLevel_) {
  if(newLevel)
    oe->pushContext();
}

NewParamLevel::~NewParamLevel() {
  if(newLevel)
    oe->popContext();
}

map<string, string> Eval::units;

Eval::Eval(vector<bfs::path> *dependencies_) : dependencies(dependencies_) {
  static bool initialized=false;
  if(!initialized) {
    // get units
    msg(Debug)<<"Build unit list for measurements."<<endl;
    bfs::path XMLDIR=installPath/"share"/"mbxmlutils"/"xml"; // use rel path if build configuration dose not work
    shared_ptr<DOMDocument> mmdoc=DOMParser::create()->parse(XMLDIR/"measurement.xml", dependencies, false);
    DOMElement *ele, *el2;
    for(ele=mmdoc->getDocumentElement()->getFirstElementChild(); ele!=nullptr; ele=ele->getNextElementSibling())
      for(el2=ele->getFirstElementChild(); el2!=nullptr; el2=el2->getNextElementSibling()) {
        if(units.find(E(el2)->getAttribute("name"))!=units.end())
          throw runtime_error(string("Internal error: Unit name ")+E(el2)->getAttribute("name")+" is defined more than once.");
        units[E(el2)->getAttribute("name")]=X()%E(el2)->getFirstTextChild()->getData();
      }
    initialized=true;
  }
};

shared_ptr<Eval> Eval::createEvaluator(const string &evalName, vector<bfs::path> *dependencies_) {
  // search if a evaluator named evalName already exits and return a new instance of it, if found
  auto it=getEvaluators().find(evalName);
  if(it!=getEvaluators().end())
    return it->second(dependencies_);

  // load the evaluator plugin named evalName
  msgStatic(Debug)<<"Loading evaluator '"<<evalName<<"'."<<endl;

  // check if a library named libmbxmlutils-eval-global-<evalName>.<ext> exists.
  // If it exists we load this library with the RTLD_GLOBAL flag (on Linux).
  // If it not exists we load a library named libmbxmlutils-eval-<evalName>.<ext> with the RTLD_LOCAL flag (on Linux).
  bfs::path libName=installPath/LIBBIN/("libmbxmlutils-eval-global-"+evalName+SHEXT);
  bool loadWithGlobalFlag=true;
  if(!bfs::exists(libName)) {
    libName=installPath/LIBBIN/("libmbxmlutils-eval-"+evalName+SHEXT);
    loadWithGlobalFlag=false;
  }

  try {
    SharedLibrary::load(canonical(libName).string(), loadWithGlobalFlag);
  }
  catch(const exception &ex) {
    throw runtime_error("Unable to load the evaluator named '"+evalName+"'.\n"
                        "System error message: "+ex.what());
  }

  // search again the evaluator named evalName and return a new instance of it or throw a error message
  it=getEvaluators().find(evalName);
  if(it!=getEvaluators().end())
    return it->second(dependencies_);
  throw runtime_error("No evaluator named '"+evalName+"' registered.");
}

Eval::~Eval() = default;

map<string, function<shared_ptr<Eval>(vector<bfs::path>*)> >& Eval::getEvaluators() {
  static map<string, function<shared_ptr<Eval>(vector<bfs::path>*)> > evaluators;
  return evaluators;
};

template<>
string Eval::cast<string>(const Value &value) const {
  return cast_string(value);
}

template<>
CodeString Eval::cast<CodeString>(const Value &value) const {
  return cast_CodeString(value);
}

template<>
double Eval::cast<double>(const Value &value) const {
  return cast_double(value);
}

template<>
int Eval::cast<int>(const Value &value) const {
  return cast_int(value);
}

template<>
vector<double> Eval::cast<vector<double> >(const Value &value) const {
  return cast_vector_double(value);
}

template<>
vector<vector<double> > Eval::cast<vector<vector<double> > >(const Value &value) const {
  return cast_vector_vector_double(value);
}

void Eval::pushContext() {
  paramStack.push(currentParam);
  importStack.push(currentImport);
}

void Eval::popContext() {
  currentParam=paramStack.top();
  paramStack.pop();
  currentImport=importStack.top();
  importStack.pop();
}

template<>
Eval::Value Eval::create<double>(const double& v) const {
  return create_double(v);
}

template<>
Eval::Value Eval::create<vector<double> >(const vector<double>& v) const {
  return create_vector_double(v);
}

template<>
Eval::Value Eval::create<vector<vector<double> > >(const vector<vector<double> >& v) const {
  return create_vector_vector_double(v);
}

template<>
Eval::Value Eval::create<string>(const string& v) const {
  return create_string(v);
}

void Eval::addParam(const string &paramName, const Value& value) {
  currentParam[paramName]=value;
}

void Eval::addParamSet(const DOMElement *e) {
  for(DOMElement *ee=e->getFirstElementChild(); ee!=nullptr; ee=ee->getNextElementSibling()) {
    if(E(ee)->getTagName()==PV%"import") {
      auto textEle=E(ee)->getFirstTextChild();
      auto text=textEle ? X()%textEle->getData() : "";
      addImport(text, ee);
    }
    else
      addParam(E(ee)->getAttribute("name"), eval(ee));
  }
}

Eval::Value Eval::eval(const DOMElement *e) {
  void *ud=e->getUserData(X()%evalValueKey);
  if(ud)
    return *static_cast<Value*>(ud);

  const DOMElement *ec;

  // check if we are evaluating a symbolic function element
  bool function=false;
  DOMNamedNodeMap *attr=e->getAttributes();
  for(int i=0; i<attr->getLength(); i++) {
    auto *a=static_cast<DOMAttr*>(attr->item(i));
    // skip xml* attributes
    if((X()%a->getName()).substr(0, 3)=="xml")
      continue;
    if(A(a)->isDerivedFrom(PV%"symbolicFunctionArgNameType")) {
      function=true;
      break;
    }
  }

  // for functions add the function arguments as parameters
  NewParamLevel newParamLevel(shared_from_this(), function);
  vector<Value> inputs;
  if(function) {
    // loop over all attributes and search for arg1name, arg2name attributes
    DOMNamedNodeMap *attr=e->getAttributes();
    for(int i=0; i<attr->getLength(); i++) {
      auto *a=static_cast<DOMAttr*>(attr->item(i));
      // skip xml* attributes
      if((X()%a->getName()).substr(0, 3)=="xml")
        continue;
      // skip all attributes not of type symbolicFunctionArgNameType
      if(!A(a)->isDerivedFrom(PV%"symbolicFunctionArgNameType"))
        continue;
      string base=X()%a->getName();
      if(base.substr(0, 3)!="arg")
        throw DOMEvalException("Internal error: a symbolicFunctionArgNameType must start with 'arg'", e);
      int baseNr = boost::lexical_cast<int>(base.substr(3));
      auto dim = E(e)->hasAttribute(base+"Dim") ? boost::lexical_cast<int>(E(e)->getAttribute(base+"Dim")) : 0;
      auto nr = E(e)->hasAttribute(base+"Nr") ? boost::lexical_cast<int>(E(e)->getAttribute(base+"Nr")) : baseNr;

      inputs.resize(max(nr, static_cast<int>(inputs.size()))); // fill new elements with default ctor
      inputs[nr-1]=createFunctionIndep(dim);
      addParam(X()%a->getValue(), inputs[nr-1]);
    }
    // check if one argument was not set. If so error
    for(auto & input : inputs)
      if(!input) // a empty object is a error (see above), since not all arg?name args were defined
        throw DOMEvalException("All argXName attributes up to the largest argument number must be specified.", e);
  }
  
  // a XML vector
  ec=E(e)->getFirstElementChildNamed(PV%"xmlVector");
  if(ec) {
    int i;
    // calculate nubmer for rows
    i=0;
    for(const DOMElement* ele=ec->getFirstElementChild(); ele!=nullptr; ele=ele->getNextElementSibling(), i++);
    // get/eval values
    vector<double> m(i);
    vector<Value> M(i);
    i=0;
    for(const DOMElement* ele=ec->getFirstElementChild(); ele!=nullptr; ele=ele->getNextElementSibling(), i++) {
      auto textEle=E(ele)->getFirstTextChild();
      auto text=textEle ? X()%textEle->getData() : "";
      if(!function) {
	Value ret=stringToValue(text, ele);
	 if(E(e)->isDerivedFrom(PV%"indexVector"))
	   convertIndex(ret, true);
        m[i]=cast<double>(ret);
      }
      else
        M[i]=stringToValue(text, ele);
    }
    if(!function)
      return handleUnit(e, create(m));
    else
      return createFunction(inputs, createFunctionDep(M));
  }
  
  // a XML matrix
  ec=E(e)->getFirstElementChildNamed(PV%"xmlMatrix");
  if(ec) {
    int i, j;
    // calculate nubmer for rows and cols
    i=0;
    for(const DOMElement* row=ec->getFirstElementChild(); row!=nullptr; row=row->getNextElementSibling(), i++);
    j=0;
    for(const DOMElement* ele=ec->getFirstElementChild()->getFirstElementChild(); ele!=nullptr; ele=ele->getNextElementSibling(), j++);
    // get/eval values
    vector<vector<double> > m(i, vector<double>(j));
    vector<vector<Value> > M(i, vector<Value>(j));
    i=0;
    for(const DOMElement* row=ec->getFirstElementChild(); row!=nullptr; row=row->getNextElementSibling(), i++) {
      j=0;
      for(const DOMElement* col=row->getFirstElementChild(); col!=nullptr; col=col->getNextElementSibling(), j++) {
        auto textEle=E(col)->getFirstTextChild();
        auto text=textEle ? X()%textEle->getData() : "";
        if(!function) {
	  Value ret=stringToValue(text, col);
	  if(E(e)->isDerivedFrom(PV%"indexMatrix"))
	    convertIndex(ret, true);
	  m[i][j]=cast<double>(ret);
	}
        else
          M[i][j]=stringToValue(text, col);
      }
    }
    if(!function)
      return handleUnit(e, create(m));
    else
      return createFunction(inputs, createFunctionDep(M));
  }
  
  // a element with a single text child (including unit conversion)
  if(!e->getFirstElementChild() &&
     E(e)->getFirstTextChild() &&
     (E(e)->isDerivedFrom(PV%"scalar") ||
      E(e)->isDerivedFrom(PV%"vector") ||
      E(e)->isDerivedFrom(PV%"matrix") ||
      E(e)->isDerivedFrom(PV%"fullEval") ||
      E(e)->isDerivedFrom(PV%"integerVector") ||
      E(e)->isDerivedFrom(PV%"indexVector") ||
      E(e)->isDerivedFrom(PV%"indexMatrix") ||
      function)
    ) {
    auto textEle=E(e)->getFirstTextChild();
    auto text=textEle ? X()%textEle->getData() : "";
    Value ret=stringToValue(text, e);
     if(E(e)->isDerivedFrom(PV%"scalar") && !valueIsOfType(ret, ScalarType))
      throw DOMEvalException("Value is not of type scalar", e);
    if(E(e)->isDerivedFrom(PV%"vector") && !valueIsOfType(ret, VectorType))
      throw DOMEvalException("Value is not of type vector", e);
    if(E(e)->isDerivedFrom(PV%"matrix") && !valueIsOfType(ret, MatrixType))
      throw DOMEvalException("Value is not of type matrix", e);
    if(E(e)->isDerivedFrom(PV%"stringFullEval") && !valueIsOfType(ret, StringType)) // also filenameFullEval
      throw DOMEvalException("Value is not of type scalar string", e);
    if(E(e)->isDerivedFrom(PV%"integerVector") && !valueIsOfType(ret, VectorType))
      throw DOMEvalException("Value is not of type vector", e);
    if(E(e)->isDerivedFrom(PV%"indexVector") && !valueIsOfType(ret, VectorType))
      throw DOMEvalException("Value is not of type vector", e);
    if(E(e)->isDerivedFrom(PV%"indexMatrix") && !valueIsOfType(ret, MatrixType))
      throw DOMEvalException("Value is not of type matrix", e);

    // handle 1 based index vectors and matrices
    if(E(e)->isDerivedFrom(PV%"indexVector") or E(e)->isDerivedFrom(PV%"indexMatrix"))
      convertIndex(ret, true);

    // add filenames to dependencies
    if(dependencies && E(e)->isDerivedFrom(PV%"filenameFullEval"))
      dependencies->push_back(E(e)->convertPath(cast<string>(ret)));

    // convert unit
    ret=handleUnit(e, ret);
  
    if(!function)
      return ret;
    else
      return createFunction(inputs, ret);
  }

  // a anyParameter element
  if(E(e)->getTagName()==PV%"anyParameter") {
    auto textEle=E(e)->getFirstTextChild();
    auto text=textEle ? X()%textEle->getData() : "";
    Value ret=stringToValue(text, e);
    return ret;
  }
  
  // rotation about x,y,z
  for(char ch='X'; ch<='Z'; ch++) {
    ec=E(e)->getFirstElementChildNamed(PV%(string("about")+ch));
    if(ec) {
      // convert
      Value angle=eval(ec);
      vector<Value> args(1);
      args[0]=angle;
      Value ret;
      try { ret=callFunction(string("rotateAbout")+ch, args); } RETHROW_AS_DOMEVALEXCEPTION(ec)
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
      vector<Value> angles(3);
      DOMElement *ele;
  
      ele=ec->getFirstElementChild();
      angles[0]=handleUnit(ec, eval(ele));
      ele=ele->getNextElementSibling();
      angles[1]=handleUnit(ec, eval(ele));
      ele=ele->getNextElementSibling();
      angles[2]=handleUnit(ec, eval(ele));
      Value ret;
      try { ret=callFunction(rotFuncName[i], angles); } RETHROW_AS_DOMEVALEXCEPTION(ec)
      return ret;
    }
  }
  
  // from file
  ec=E(e)->getFirstElementChildNamed(PV%"fromFile");
  if(ec) {
    Value fileName=stringToValue(E(ec)->getAttribute("href"), ec, false);
    if(dependencies)
      dependencies->push_back(E(e)->convertPath(cast<string>(fileName)));

    // restore current dir on exit and change current dir
    PreserveCurrentDir preserveDir;
    bfs::path chdir=E(e)->getOriginalFilename().parent_path();
    if(!chdir.empty())
      bfs::current_path(chdir);

    Value ret;
    vector<Value> args(1);
    args[0]=fileName;
    try { ret=callFunction("load", args); } RETHROW_AS_DOMEVALEXCEPTION(ec)
    handleUnit(e, ret);
    return ret;
  }

  // a element of type PV%"script": return a string containing a special coded form
  // of all current parameters
  if(!e->getFirstElementChild() &&
     E(e)->getFirstTextChild() &&
     E(e)->isDerivedFrom(PV%"script")) {
    // convert all currentParam to string
    string ret;
    for(auto &p: currentParam) {
      string type;
      if(valueIsOfType(p.second, FunctionType)) // skip functions
        continue;
      else if(valueIsOfType(p.second, ScalarType))
        type="scalar";
      else if(valueIsOfType(p.second, VectorType))
        type="vector";
      else if(valueIsOfType(p.second, MatrixType))
        type="matrix";
      else if(valueIsOfType(p.second, StringType))
        type="string";
      ret+=type+":"+p.first+"="+cast<CodeString>(p.second)+"\n";
    }
    return create(ret);
  }
  
  // unknown element: throw
  throw DOMEvalException("Dont know how to evaluate this element", e);
}

Eval::Value Eval::eval(const xercesc::DOMAttr *a) {
  bool fullEval;
  if(A(a)->isDerivedFrom(PV%"fullEval"))
    fullEval=true;
  else if(A(a)->isDerivedFrom(PV%"partialEval"))
    fullEval=false;
  else
    throw DOMEvalException("Unknown XML attribute type", a);

  // evaluate attribute fully
  if(fullEval) {
    Value ret=stringToValue(X()%a->getValue(), a->getOwnerElement());
    if(A(a)->isDerivedFrom(PV%"floatFullEval")) {
      if(!valueIsOfType(ret, ScalarType))
        throw DOMEvalException("Value is not of type scalar float", a);
    }
    else if(A(a)->isDerivedFrom(PV%"stringFullEval")) {
      if(!valueIsOfType(ret, StringType)) // also filenameFullEval
        throw DOMEvalException("Value is not of type scalar string", a);
    }
    else if(A(a)->isDerivedFrom(PV%"integerFullEval")) {
      try { cast<int>(ret); } RETHROW_AS_DOMEVALEXCEPTION(a);
    }
    else if(A(a)->isDerivedFrom(PV%"booleanFullEval")) {
      int value;
      try { value=cast<int>(ret); } RETHROW_AS_DOMEVALEXCEPTION(a);
      if(value!=0 && value!=1)
        throw DOMEvalException("Value is not of type scalar boolean", a);
    }
    else if(A(a)->isDerivedFrom(PV%"indexFullEval")) {
      try { cast<int>(ret); } RETHROW_AS_DOMEVALEXCEPTION(a);
      convertIndex(ret, true);
    }
    else
      throw DOMEvalException("Unknown XML attribute type for evaluation", a);

    // add filenames to dependencies
    if(dependencies && A(a)->isDerivedFrom(PV%"filenameFullEval"))
      dependencies->push_back(E(a->getOwnerElement())->convertPath(cast<string>(ret)));

    return ret;
  }
  // evaluate attribute partially
  else {
    Value ret;
    string s=partialStringToString(X()%a->getValue(), a->getOwnerElement());
    if(A(a)->isDerivedFrom(PV%"varnamePartialEval")) { // also symbolicFunctionArgNameType
      if(s.length()<1)
        throw DOMEvalException("A variable name must consist of at least 1 character", a);
      if(!(s[0]=='_' || ('a'<=s[0] && s[0]<='z') || ('A'<=s[0] && s[0]<='Z')))
        throw DOMEvalException("A variable name start with _, a-z or A-Z", a);
      for(size_t i=1; i<s.length(); i++)
        if(!(s[i]=='_' || ('a'<=s[i] && s[i]<='z') || ('A'<=s[i] && s[i]<='Z') || ('0'<=s[i] && s[i]<='9')))
          throw DOMEvalException("Only the characters _, a-z, A-Z and 0-9 are allowed for variable names", a);
      ret=create(s);
    }
    else if(A(a)->isDerivedFrom(PV%"floatPartialEval"))
      try { ret=create(boost::lexical_cast<double>(s)); }
      catch(const boost::bad_lexical_cast &) { throw DOMEvalException("Value is not of type scalar float", a); }
    else if(A(a)->isDerivedFrom(PV%"stringPartialEval")) // also filenamePartialEval
      try { ret=create(s); }
      catch(const boost::bad_lexical_cast &) { throw DOMEvalException("Value is not of type scalar string", a); }
    else if(A(a)->isDerivedFrom(PV%"integerPartialEval"))
      try { ret=create<double>(boost::lexical_cast<int>(s)); }
      catch(const boost::bad_lexical_cast &) { throw DOMEvalException("Value is not of type scalar integer", a); }
    else if(A(a)->isDerivedFrom(PV%"booleanPartialEval"))
      try { ret=create<double>(boost::lexical_cast<bool>(s)); }
      catch(const boost::bad_lexical_cast &) { throw DOMEvalException("Value is not of type scalar boolean", a); }
    else if(A(a)->isDerivedFrom(PV%"indexPartialEval")) {
      try { ret=create<double>(boost::lexical_cast<int>(s)); }
      catch(const boost::bad_lexical_cast &) { throw DOMEvalException("Value is not of type scalar integer", a); }
      convertIndex(ret, true);
    }
    else
      throw DOMEvalException("Unknown XML attribute type for evaluation", a);

    // add filenames to dependencies
    if(dependencies && A(a)->isDerivedFrom(PV%"filenamePartialEval"))
      dependencies->push_back(E(a->getOwnerElement())->convertPath(s));

    return ret;
  }
}

Eval::Value Eval::eval(const string &str, const DOMElement *e, bool skipRet) {
  return fullStringToValue(str, e, skipRet);
}

Eval::Value Eval::handleUnit(const xercesc::DOMElement *e, const Value &ret) {
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
  // handle common unit conversions very fast (without evaluation)
  if(eqn=="value")
    return ret;
  // all other conversion must be processed using the evaluator
  NewParamLevel newParamLevel(shared_from_this(), true);
  addParam("value", ret);
  return stringToValue(eqn, e);
}

string Eval::partialStringToString(const string &str, const DOMElement *e) const {
  namespace qi = boost::spirit::qi;
  namespace phx = boost::phoenix;
  using It = string::const_iterator;

  static bool init=false;
  vector<char> start;
  static qi::rule<It, pair<vector<char>, vector<char>>()> exprText;
  if(!init) {
    init=true;
    static qi::symbols<char, char> esc;
    esc.add("\\{", '{');
    esc.add("\\}", '}');
    exprText = ('{' >> *(esc | (qi::char_ - '}')) >> '}' >> *(qi::char_ - '{'))
               [qi::_val=phx::construct<pair<vector<char>, vector<char>>>(qi::_1, qi::_2)];
  }
  qi::rule<It, vector<pair<vector<char>, vector<char>>>()> name;
  name = (*(qi::char_ - '{'))[phx::ref(start)=qi::_1] >> (*exprText)[qi::_val=qi::_1] >> qi::eoi;
  vector<pair<vector<char>, vector<char>>> parsedContent;
  if(!qi::parse(str.begin(), str.end(), name, parsedContent))
    throw runtime_error("Cannot parse the partial evaluation string:\n"+str);
  // init s with the text before the first expression
  string s(start.begin(), start.end());
  for(auto &exprName: parsedContent) {
    // eval expression
    Value ret=fullStringToValue(string(exprName.first.begin(), exprName.first.end()), e);
    string subst;
    try {
      if(valueIsOfType(ret, ScalarType)) {
        double d=cast<double>(ret);
        int i;
        if(tryDouble2Int(d, i))
          subst=fmatvec::toString(i);
        else
          subst=fmatvec::toString(d);
      }
      else if(valueIsOfType(ret, StringType))
        subst=cast<string>(ret);
      else
        throw runtime_error("Partial evaluations can only be of type scalar or string.");
    } RETHROW_AS_DOMEVALEXCEPTION(e);
    s+=subst;
    // append text after expression^
    s+=string(exprName.second.begin(), exprName.second.end());
  }
  return s;
}

Eval::Value Eval::stringToValue(const string &str, const DOMElement *e, bool fullEval) const {
  if(fullEval)
    return fullStringToValue(str, e);
  else
    return create(partialStringToString(str, e));
}

CodeString Eval::cast_CodeString(const Value &value) const {
  ostringstream ret;
  ret.precision(numeric_limits<double>::digits10+1);
  if(valueIsOfType(value, ScalarType)) {
    ret<<cast<double>(value);
    return ret.str();
  }
  if(valueIsOfType(value, VectorType)) {
    vector<double> v=cast<vector<double> >(value);
    ret<<("[");
    for(int i=0; i<v.size(); ++i) {
      ret<<v[i];
      if(i!=v.size()-1) ret<<"; ";
    }
    ret<<"]";
    return ret.str();
  }
  else if(valueIsOfType(value, MatrixType)) {
    vector<vector<double> > m=cast<vector<vector<double> > >(value);
    ret<<("[");
    for(int r=0; r<m.size(); ++r) {
      for(int c=0; c<m[r].size(); ++c) {
        ret<<m[r][c];
        if(c!=m[r].size()-1) ret<<",";
      }
      if(r!=m.size()-1) ret<<"; ";
    }
    ret<<"]";
    return ret.str();
  }
  else if(valueIsOfType(value, StringType)) {
    ret<<"'"<<cast<string>(value)<<"'";
    return ret.str();
  }
  else if(valueIsOfType(value, FunctionType)) {
    return serializeFunction(value);
  }
  else
    return getStringRepresentation(value);
}

int Eval::cast_int(const Value &value) const {
  double d=cast<double>(value);
  int i;
  if(!tryDouble2Int(d, i))
    throw runtime_error("Cannot cast this value to int.");
  return i;
}

string Eval::getStringRepresentation(const Value &x) const {
  throw runtime_error("Cannot cast this value to a evaluator specific string representation.");
}

void Eval::addStaticDependencies(const DOMElement *e) const {
  if(!dependencies)
    return;
  for(auto & mbxmlutilsStaticDependencie : mbxmlutilsStaticDependencies) {
    bfs::path path=E(e)->convertPath(mbxmlutilsStaticDependencie);
    dependencies->push_back(path);
  }
}

void Eval::setValue(DOMElement *e, const Value &v) {
  e->setUserData(X()%evalValueKey, new Value(v), &valueUserDataHandler);
}

void Eval::printEvaluatorMsg(const std::ostringstream &str, MsgType msgType) {
  if(!str.str().empty()) {
      std::string msg=str.str();
    trim_right_if(msg, boost::is_any_of(" \n"));
    fmatvec::Atom::msgStatic(msgType)<<"Evaluator output: "<<msg<<std::endl;
  }
}

} // end namespace MBXMLUtils
