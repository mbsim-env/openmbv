#ifndef _MBXMLUTILS_OCTEVAL_H_
#define _MBXMLUTILS_OCTEVAL_H_

#include <string>
#include <stack>
#ifdef HAVE_UNORDERED_MAP
#  include <unordered_map>
#else
#  include <map>
#  define unordered_map map
#endif
#include <octave/oct.h>
#include "mbxmlutilstinyxml/casadiXML.h"

#define MBXMLUTILSPVNS "{http://openmbv.berlios.de/MBXMLUtils/physicalvariable}"

namespace MBXMLUtils {

class TiXmlElement;

class OctEvalException {
  public:
    OctEvalException(const std::string &msg_, const TiXmlElement *e, const std::string &attrName=std::string());
    void print() const;
  protected:
    std::vector<std::string> msg;
};

class OctEval {
  public:
    //! Known type for the "ret" variable
    enum ValueType {
      ScalarType,
      VectorType,
      MatrixType,
      StringType,
      ScalarFunction,
      VectorFunction,
      MatrixFunction
    };

    //! ctor
    OctEval();
    //! dtor
    ~OctEval();

    //! add a octave value to the current parameters
    void addParam(const std::string &paramName, const octave_value& value);
    //! add all parameters from element e, the parameters may depend on each other
    void addParamSet(const TiXmlElement *e);

    //! push the current parameters to a internal stack
    void pushParams();
    //! overwrite the current parameter with the top level set on the internal stack
    void popParams();
    
    //! Evaluate the element e using the current parameters returning the resulting octave value.
    //! Handle the attribute name attrName if given or handle the element e.
    //! If attrName if given evaluate it fully if fullEval is true or partially.
    //! The result of a partially evaluation is returned as a octave string even so it is not really a string.
    octave_value eval(const TiXmlElement *e, const std::string &attrName=std::string(), bool fullEval=true);

    // helper functions

    //! cast octave value to T
    template<typename T>
    static T cast(const octave_value &value);

  protected:

    //! create octave value of type SXMatrix
    octave_value createSXMatrix(const std::string &name, int dim1, int dim2);

    //! create octave value of type SXFunction
    octave_value createSXFunction(const Cell &inputs, const octave_value &output);

    octave_value stringToOctValue(const std::string &str, const TiXmlElement *e) const;

    octave_value octaveCasADiSXFunction(Cell inputs, const std::vector<std::vector<octave_value> > &output, bool eleMustBeScalar);

    //! get the type of value
    static ValueType getType(const octave_value &value);

    // map of the current parameters
    std::unordered_map<std::string, octave_value> currentParam;

    // stack of parameters
    std::stack<std::unordered_map<std::string, octave_value> > paramStack;

    static int initCount;

    static std::map<std::string, std::string> units;

    octave_value casadiOctValue;
};

} // end namespace MBXMLUtils

#endif
