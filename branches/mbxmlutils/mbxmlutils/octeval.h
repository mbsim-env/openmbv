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
#include <boost/intrusive_ptr.hpp>
#include <octave/parse.h>

#define MBXMLUTILSPVNS "{http://openmbv.berlios.de/MBXMLUtils/physicalvariable}"

namespace MBXMLUtils {

class TiXmlElement;

// A class to block/unblock stderr or stdout. Block in called in the ctor, unblock in the dtor
template<int T>
class Block {
  public:
    Block(std::ostream &str_) : str(str_) {
      if(disableCount==0)
        orgcxxx=str.rdbuf(0);
      disableCount++;
    }
    ~Block() {
      disableCount--;
      if(disableCount==0)
        str.rdbuf(orgcxxx);
    }
  private:
    std::ostream &str;
    static std::streambuf *orgcxxx;
    static int disableCount;
};
template<int T> std::streambuf *Block<T>::orgcxxx;
template<int T> int Block<T>::disableCount=0;
#define BLOCK_STDOUT Block<1> mbxmlutils_dummy_blockstdout(std::cout);
#define BLOCK_STDERR Block<2> mbxmlutils_dummy_blockstderr(std::cerr);

class OctEvalException {
  public:
    OctEvalException(const std::string &msg_, const TiXmlElement *e, const std::string &attrName=std::string());
    void print() const;
  protected:
    std::vector<std::string> msg;
};

/* Octave expression evaluator and converter. */
class OctEval {
  public:
    //! Constructor.
    OctEval();
    //! Destructor.
    ~OctEval();

    //! Add a octave value to the current parameters.
    void addParam(const std::string &paramName, const octave_value& value);
    //! Add all parameters from XML element e. The parameters may depend on each other.
    void addParamSet(const TiXmlElement *e);

    //! Push the current parameters to a internal stack.
    void pushParams();
    //! Overwrite the current parameter with the top level set from the internal stack.
    void popParams();
    
    //! Evaluate the XML element e using the current parameters returning the resulting octave value.
    //! Handle the attribute value named attrName, or if not given handle the XML text node child of e.
    //! If attrName if given evaluate it "fully" if fullEval is true or "partially".
    //! The result of a "partially" evaluation is returned as a octave string even so it is not really a string.
    octave_value eval(const TiXmlElement *e, const std::string &attrName=std::string(), bool fullEval=true);

    /*! Cast the octave value value to type T.
     * Possible combinations of allowed octave value types and template types T are listed in the
     * following table. If a combination is not allowed a exception is thrown.
     * <table>
     * <tr><td></td><th colspan="5">Octave Type of value</th></tr>
     * <tr><th>Template Type T</th><th>scalar</th><th>vector</th><th>matrix</th><th>SWIG</th><th>SWIG CasADi::SXFunction</th></tr>
     * <tr><th>double</th><td>O</td><td></td><td></td><td></td><td></td></tr>
     * <tr><th>vector&lt;double&gt;</th><td>O</td><td>O</td><td></td><td></td><td></td></tr>
     * <tr><th>vector&lt;vector&lt;double&gt;&bnsp;&gt;</th><td>O</td><td>O</td><td>O</td><td></td><td></td></tr>
     * <tr><th>string</th><td>O (e.g. '5')</td><td>O (e.g. '[3;5]')</td><td>O (e.g. '[1,2;3,4]')</td><td></td><td></td></tr>
     * <tr><th>TiXmlElement*</th><td></td><td></td><td></td><td></td><td>O</td></tr>
     * <tr><th>SWIG_Type*</th><td></td><td></td><td></td><td>O (no type check done)</td><td>O (no type check done)</td></tr>
     * <tr><th>SWIG_Type</th><td></td><td></td><td></td><td>O (no type check done)</td><td>O (no type check done)</td></tr>
     * <tr><th>CasADi::SXMatrix</th><td>O</td><td>O</td><td>O</td><td></td><td></td></tr>
     * </table>
     */
    template<typename T>
    static T cast(const octave_value &value);

  protected:
    //! Known type for the "ret" variable
    enum ValueType {
      ScalarType,
      VectorType,
      MatrixType,
      StringType,
      SXMatrixType,
      SXFunctionType
    };

    //! create octave value of CasADi type name. Created using the default ctor.
    octave_value createCasADi(const std::string &name);

    // clone the octave value to type T
    template<typename T>
    static T cloneAs(const octave_value &value);

    octave_value stringToOctValue(const std::string &str, const TiXmlElement *e) const;

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

// default: not cloneable
template<typename T>
T OctEval::cloneAs(const octave_value &value) {
  throw std::runtime_error("Cannot clone this type of octave_value.");
}

// Helper class which convert a void* to T* or T.
template<typename T>
struct Ptr {
  static T cast(void *ptr) { return *static_cast<T*>(ptr); }
};
template<typename T>
struct Ptr<T*> {
  static T* cast(void *ptr) { return static_cast<T*>(ptr); }
};

// cast octave value to swig object ptr or swig object copy
template<typename T>
T OctEval::cast(const octave_value &value) {
  // get the casadi pointer: octave returns a 64bit integer which represents the pointer
  static octave_function *swig_this=symbol_table::find_function("swig_this").function_value(); // get ones a pointer to swig_this for performance reasons
  octave_value swigThis;
  {
    BLOCK_STDERR
    swigThis=feval(swig_this, value, 1)(0);
  }
  // try to clone it if value is not a casadi object
  if(error_state!=0) {
    error_state=0;
    return cloneAs<T>(value);
  }
  void *swigPtr=reinterpret_cast<void*>(swigThis.uint64_scalar_value().value());
  // convert the void pointer to the correct casadi type
  return Ptr<T>::cast(swigPtr);
}

} // end namespace MBXMLUtils

#endif
