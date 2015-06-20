#include <config.h>
#include "mbxmlutils/eval.h"
#include "mbxmlutils/octeval.h"
#include <xercesc/dom/DOMElement.hpp>
#include <mbxmlutilshelper/dom.h>

using namespace std;
using namespace casadi;
using namespace boost;
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

Eval::Eval(vector<bfs::path> *dependencies_) : dependencies(dependencies_) {
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
xercesc::DOMElement* Eval::cast<xercesc::DOMElement*>(const shared_ptr<void> &value, xercesc::DOMDocument *doc) {
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

void Eval::addParamSet(const xercesc::DOMElement *e) {
  for(const xercesc::DOMElement *ee=e->getFirstElementChild(); ee!=NULL; ee=ee->getNextElementSibling()) {
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

} // end namespace MBXMLUtils
