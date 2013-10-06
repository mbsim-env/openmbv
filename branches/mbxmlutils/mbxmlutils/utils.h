#ifndef _MBXMLUTILS_UTILS_H_
#define _MBXMLUTILS_UTILS_H_

#include <string>
#include <map>
#include <vector>
#include <stack>
#ifdef HAVE_UNORDERED_MAP
#  include <unordered_map>
#else
#  include <map>
#  define unordered_map map
#endif
#include <octave/oct.h>

#define MBXMLUTILSPVNS "{http://openmbv.berlios.de/MBXMLUtils/physicalvariable}"

namespace MBXMLUtils {

class TiXmlElement;

class OctaveEvaluator {
  public:
    enum ValueType {
      ArbitraryType,
      ScalarType,
      VectorType,
      MatrixType,
      StringType
    };

    OctaveEvaluator();
    void octaveAddParam(const std::string &paramName, const octave_value& value, bool useCache=true);
    void octaveAddParam(const std::string &paramName, double value, bool useCache=true);
    void octavePushParams();
    void octavePopParams();
    TiXmlElement* octaveEvalRet(std::string str, TiXmlElement *e=NULL, bool useCache=true, std::vector<std::pair<std::string, int> > *arg=NULL);
    static void checkType(const octave_value& val, ValueType expectedType);
    static std::string octaveGetRet(ValueType expectedType=ArbitraryType);
    static double octaveGetDoubleRet();
    static octave_value& octaveGetOctaveValueRet();
    void fillParam(TiXmlElement *e, bool useCache=true);
    void saveAndClearCurrentParam();
    void restoreCurrentParam();

    static void initialize();
    static void terminate();
    static void addPath(const std::string &path);

    void setUnits(const std::map<std::string, std::string> &units_) { units=units_; }
    void eval(TiXmlElement *e, bool useCache=true);

  protected:
    // map of the current parameters
    std::map<std::string, octave_value> currentParam, savedCurrentParam;
    // stack of parameters
    std::stack<std::map<std::string, octave_value> > paramStack;
    std::vector<int> currentParamHash;

    std::unordered_map<std::string, octave_value> cache;

    std::map<std::string, std::string> units;
};

} // end namespace MBXMLUtils

#endif
