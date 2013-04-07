#ifndef _MBXMLUTILS_UTILS_H_
#define _MBXMLUTILS_UTILS_H_

#include <string>
#include <map>
#include <vector>
#include <stack>
#include <octave/oct.h>
#ifdef HAVE_UNORDERED_MAP
#  include <unordered_map>
#else
#  include <map>
#  define unordered_map map
#endif

class TiXmlElement;

namespace MBXMLUtils {

class OctaveEvaluator {
  public:
    enum ValueType {
      ArbitraryType,
      ScalarType,
      VectorType,
      MatrixType,
      StringType
    };
    struct Param {
      Param(std::string n, std::string eq, TiXmlElement *e) : name(n), equ(eq), ele(e) {}
      std::string name, equ;
      TiXmlElement *ele;
    };

    OctaveEvaluator();
    void octaveAddParam(const std::string &paramName, const octave_value& value, bool useCache=true);
    void octavePushParams();
    void octavePopParams();
    void octaveEvalRet(std::string str, TiXmlElement *e=NULL, bool useCache=true);
    void checkType(const octave_value& val, ValueType expectedType);
    std::string octaveGetRet(ValueType expectedType=ArbitraryType);
    int fillParam(TiXmlElement *e, bool useCache=true);
    int fillParam(std::vector<Param> param, bool useCache=true);
    void saveAndClearCurrentParam();
    void restoreCurrentParam();

  protected:
    // map of the current parameters
    std::map<std::string, octave_value> currentParam, savedCurrentParam;
    // stack of parameters
    std::stack<std::map<std::string, octave_value> > paramStack;
    std::vector<int> currentParamHash;

    std::unordered_map<std::string, octave_value> cache;
};

} // end namespace MBXMLUtils

#endif
