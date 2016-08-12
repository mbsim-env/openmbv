#include "config.h"
#include "xmlflateval.h"
#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/lexical_cast.hpp>

using namespace boost::filesystem;
using namespace std;
using namespace xercesc;

namespace XERCES_CPP_NAMESPACE {
  class DOMElement;
}

namespace MBXMLUtils {

// register this evaluator in the object factory
XMLUTILS_EVAL_REGISTER(XMLFlatEval)

// ctor
XMLFlatEval::XMLFlatEval(std::vector<path> *dependencies_) : Eval(dependencies_) {
}

// dtor
XMLFlatEval::~XMLFlatEval() {
}

// virtual functions

void XMLFlatEval::addImport(const string &code, const DOMElement *e, bool deprecated) {
  throw runtime_error("addImport not possible.");
}

bool XMLFlatEval::valueIsOfType(const Value &value, ValueType type) const {
  switch(type) {
    case ScalarType: try { cast<double>(value); return true; } catch(...) { return false; };
    case VectorType: try { cast<vector<double> >(value); return true; } catch(...) { return false; };
    case MatrixType: try { cast<vector<vector<double> > >(value); return true; } catch(...) { return false; };
    case StringType: try { cast<string>(value); return true; } catch(...) { return false; };
    case FunctionType: return false;
  }
  return false;
}

map<path, pair<path, bool> >& XMLFlatEval::requiredFiles() const {
  static map<path, pair<path, bool> > ret;
  return ret;
}

Eval::Value XMLFlatEval::createSwigByTypeName(const string &typeName) const {
  throw runtime_error("createSwig<T> not possible.");
}

Eval::Value XMLFlatEval::callFunction(const string &name, const vector<Value>& args) const {
  throw runtime_error("callFunction not possible.");
}

Eval::Value XMLFlatEval::fullStringToValue(const string &str, const DOMElement *e) const {
  return make_shared<string>(str);
}

void* XMLFlatEval::getSwigThis(const Value &value) const {
  throw runtime_error("getSwigThis not possible.");
}

string XMLFlatEval::getSwigType(const Value &value) const {
  throw runtime_error("getSwigThis not possible.");
}

double XMLFlatEval::cast_double(const Value &value) const {
  string *v=static_cast<string*>(boost::get<shared_ptr<void> >(value).get());
  return boost::lexical_cast<double>(*v);
}

vector<double> XMLFlatEval::cast_vector_double(const Value &value) const {
  string valueStr=*static_cast<string*>(boost::get<shared_ptr<void> >(value).get());
  boost::algorithm::trim(valueStr);
  if(valueStr[0]!='[') valueStr="["+valueStr+"]"; // surround with [ ] if not already done
  if(valueStr[valueStr.size()-1]!=']')
    throw runtime_error("Cannot cast to vector.");
  // add some spaces
  boost::algorithm::replace_all(valueStr, "[", "[ ");
  boost::algorithm::replace_all(valueStr, "]", " ]");
  boost::algorithm::replace_all(valueStr, ";", " ; ");
  boost::algorithm::replace_all(valueStr, "\n", " ; ");
  istringstream str(valueStr);
  string s;
  str>>s; // first token [
  vector<double> v;
  while(1) {
    str>>s; // read next token
    if(s==";") // on ; read next
      continue;
    else if(s=="]") // on ] exit
      break;
    else // else push double to vector
      v.push_back(boost::lexical_cast<double>(s));
  }
  return v;
}

vector<vector<double> > XMLFlatEval::cast_vector_vector_double(const Value &value) const {
  string valueStr=*static_cast<string*>(boost::get<shared_ptr<void> >(value).get());
  boost::algorithm::trim(valueStr);
  if(valueStr[0]!='[') valueStr="["+valueStr+"]"; // surround with [] if not already done
  if(valueStr[valueStr.size()-1]!=']')
    throw runtime_error("Cannot cast to matrix.");
  // add some spaces
  boost::algorithm::replace_all(valueStr, "[", "[ ");
  boost::algorithm::replace_all(valueStr, "]", " ]");
  boost::algorithm::replace_all(valueStr, ",", " , ");
  boost::algorithm::replace_all(valueStr, ";", " ; ");
  boost::algorithm::replace_all(valueStr, "\n", " ; ");
  istringstream str(valueStr);
  string s;
  str>>s; // first token
  vector<vector<double> > m;
  m.push_back(vector<double>());
  while(1) {
    str>>s; // read next token
    if(s==";") // on ; new row
      m.push_back(vector<double>());
    else if(s==",") // on , read next
      continue;
    else if(s=="]") // on ] exit
      break;
    else // else push double to vector
      (--m.end())->push_back(boost::lexical_cast<double>(s));
  }
  return m;
}

string XMLFlatEval::cast_string(const Value &value) const {
  string valueStr=*static_cast<string*>(boost::get<shared_ptr<void> >(value).get());
  boost::algorithm::trim(valueStr);
  if(valueStr[0]!='\'' || valueStr[valueStr.size()-1]!='\'')
    throw runtime_error("Cannot convert to string.");
  return valueStr.substr(1, valueStr.size()-2);
}

Eval::Value XMLFlatEval::create_double(const double& v) const {
  return make_shared<string>(to_string(v));
}

Eval::Value XMLFlatEval::create_vector_double(const vector<double>& v) const {
  string str("[");
  for(int i=0; i<v.size(); ++i) {
    str+=to_string(v[i]);
    if(i!=v.size()-1) str+=";";
  }
  str+="]";
  return make_shared<string>(str);
}

Eval::Value XMLFlatEval::create_vector_vector_double(const vector<vector<double> >& v) const {
  string str("[");
  for(int r=0; r<v.size(); ++r) {
    for(int c=0; c<v[r].size(); ++c) {
      str+=to_string(v[r][c]);
      if(c!=v[r].size()-1) str+=",";
    }
    if(r!=v.size()-1) str+=";";
  }
  str+="]";
  return make_shared<string>(str);
}

Eval::Value XMLFlatEval::create_string(const string& v) const {
  return make_shared<string>("'"+v+"'");
}

} // end namespace MBXMLUtils
