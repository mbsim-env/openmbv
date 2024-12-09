#include "config.h"
#include "xmlflateval.h"
#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <boost/lexical_cast.hpp>
#include <fmatvec/toString.h>
#include <fmatvec/ast.h>
#include <regex>

using namespace boost::filesystem;
using namespace std;
using namespace xercesc;

namespace XERCES_CPP_NAMESPACE {
  class DOMElement;
}

namespace MBXMLUtils {

// register this evaluator in the object factory
MBXMLUTILS_EVAL_REGISTER(XMLFlatEval)

// ctor
XMLFlatEval::XMLFlatEval(vector<path> *dependencies_) : Eval(dependencies_) {
}

// dtor
XMLFlatEval::~XMLFlatEval() = default;

// virtual functions

Eval::Value XMLFlatEval::createFunctionIndep(int dim) const {
  return make_shared<string>();
}

void XMLFlatEval::addImport(const string &code, const DOMElement *e, const string &action) {
  throw runtime_error("addImport not possible.");
}

bool XMLFlatEval::valueIsOfType(const Value &value, ValueType type) const {
  string valueStr=*static_cast<string*>(value.get());
  boost::trim(valueStr);
  switch(type) {
    case ScalarType:
      if(valueStr=="true") return true; // "true" is a scalar
      if(valueStr=="false") return true; // "false" is a scalar
      double d;
      return boost::conversion::try_lexical_convert(valueStr, d); // any value e.g. -6.73 is a scalar
    case VectorType:
      if(valueIsOfType(value, ScalarType)) return true;
      if(valueStr[0]!='[') return false; // if not starting with "[" it NOT a vector
      // it is starting with "["
      // remove the valid starting "[" and trailing "]"
      valueStr=valueStr.substr(1, valueStr.length()-2);
      boost::trim(valueStr);
      // remove the valid "new row" regex (" *; *")
      static std::regex re(" *; *");
      valueStr=regex_replace(valueStr, re, "");
      // if now some invalid "new col" (" " or ",") is contained than its NOT a vector
      if(valueStr.find(' ')!=string::npos) return false;
      if(valueStr.find(',')!=string::npos) return false;
      return true;
    case MatrixType:
      if(valueIsOfType(value, ScalarType)) return true;
      return valueStr[0]=='['; // everything starting with "[" is s matrix
    case StringType:
      return valueStr[0]=='\'' || valueStr[0]=='"'; // everything starting with "'" or with '"' is a string
    case FunctionType: {
      return valueStr.substr(0,2)=="f("; // everything starting with "f(" is a function
    }
  }
  return false;
}

map<path, pair<path, bool> >& XMLFlatEval::requiredFiles() const {
  static map<path, pair<path, bool> > ret;
  return ret;
}

Eval::Value XMLFlatEval::callFunction(const string &name, const vector<Value>& args) const {
  throw runtime_error("callFunction not possible.");
}

Eval::Value XMLFlatEval::fullStringToValue(const string &str, const DOMElement *e, bool skipRet) const {
  return make_shared<string>(str);
}

double XMLFlatEval::cast_double(const Value &value) const {
  auto *v=static_cast<string*>(value.get());
  auto vStr=boost::algorithm::trim_copy(*v);
  if(vStr=="true") return 1;
  if(vStr=="false") return 0;
  return boost::lexical_cast<double>(vStr);
}

vector<double> XMLFlatEval::cast_vector_double(const Value &value) const {
  string valueStr=*static_cast<string*>(value.get());
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
  str.exceptions(ios::failbit | ios::badbit);
  string s;
  str>>s; // first token [
  vector<double> v;
  while(true) {
    str>>s; // read next token
    if(s==";") // on ; read next
      continue;
    else if(s=="]") // on ] exit
      break;
    else { // else push double to vector
      if(s=="true") v.push_back(1);
      else if(s=="false") v.push_back(0);
      else v.push_back(boost::lexical_cast<double>(s));
    }
  }

  // check end of stream
  if(!str.eof())
    str>>ws;
  if(!str.eof())
    throw runtime_error("Input not fully read.");

  return v;
}

vector<vector<double> > XMLFlatEval::cast_vector_vector_double(const Value &value) const {
  string valueStr=*static_cast<string*>(value.get());
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
  str.exceptions(ios::failbit | ios::badbit);
  string s;
  str>>s; // first token
  vector<vector<double> > m;
  m.emplace_back();
  while(true) {
    str>>s; // read next token
    if(s==";") // on ; new row
      m.emplace_back();
    else if(s==",") // on , read next
      continue;
    else if(s=="]") // on ] exit
      break;
    else { // else push double to vector
      if(s=="true") (--m.end())->push_back(1);
      else if(s=="false") (--m.end())->push_back(0);
      (--m.end())->push_back(boost::lexical_cast<double>(s));
    }
  }

  // check end of stream
  if(!str.eof())
    str>>ws;
  if(!str.eof())
    throw runtime_error("Input not fully read.");

  return m;
}

string XMLFlatEval::cast_string(const Value &value) const {
  string valueStr=*static_cast<string*>(value.get());
  boost::algorithm::trim(valueStr);
  if((valueStr[0]!='\'' && valueStr[0]!='"') ||
     (valueStr[valueStr.size()-1]!='\'' && valueStr[valueStr.size()-1]!='"'))
    throw runtime_error("Cannot convert to string.");
  return valueStr.substr(1, valueStr.size()-2);
}

Eval::Value XMLFlatEval::create_double(const double& v) const {
  return make_shared<string>(fmatvec::toString(v));
}

Eval::Value XMLFlatEval::create_vector_double(const vector<double>& v) const {
  string str("[");
  for(int i=0; i<v.size(); ++i) {
    str+=fmatvec::toString(v[i]);
    if(i!=v.size()-1) str+=";";
  }
  str+="]";
  return make_shared<string>(str);
}

Eval::Value XMLFlatEval::create_vector_vector_double(const vector<vector<double> >& v) const {
  string str("[");
  for(int r=0; r<v.size(); ++r) {
    for(int c=0; c<v[r].size(); ++c) {
      str+=fmatvec::toString(v[r][c]);
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

Eval::Value XMLFlatEval::createFunctionDep(const vector<Value>& v) const {
  return make_shared<string>();
}

Eval::Value XMLFlatEval::createFunctionDep(const vector<vector<Value> >& v) const {
  return make_shared<string>();
}

Eval::Value XMLFlatEval::createFunction(const vector<Value> &indeps, const Value &dep) const {
  return dep;
}

string XMLFlatEval::serializeFunction(const Value &x) const {
  return *static_cast<string*>(x.get());
}

} // end namespace MBXMLUtils
