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
#include <octave/parse.h>
#include <boost/filesystem.hpp>

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
    OctEvalException(const std::string &msg_, const TiXmlElement *e);
    void print() const;
  protected:
    std::vector<std::string> msg;
};

class OctEval;

//! Create a new parameter level for a octave evaluator which is automatically resetted if the scope of this object is left.
class NewParamLevel {
  public:
    //! Create a new parameter level in the octave evaluator oe_
    NewParamLevel(OctEval &oe_, bool newLevel_=true);
    //! Reset to the previous parameter level
    ~NewParamLevel();
  protected:
    OctEval &oe;
    bool newLevel;
};

/*! Octave expression evaluator and converter. */
class OctEval {
  public:
    friend class NewParamLevel;

    //! Known type for the "ret" variable
    enum ValueType {
      ScalarType,
      VectorType,
      MatrixType,
      StringType,
      SXMatrixType,
      SXFunctionType
    };

    //! Constructor.
    OctEval();
    //! Destructor.
    ~OctEval();

    //! Add a octave value to the current parameters.
    void addParam(const std::string &paramName, const octave_value& value);
    //! Add all parameters from XML element e. The parameters may depend on each other.
    void addParamSet(const TiXmlElement *e);

    //! Add dir to octave search path
    void addPath(const boost::filesystem::path &dir);
    
    //! Evaluate the XML element e using the current parameters returning the resulting octave value.
    //! Handle the attribute value named attrName, or if not given handle the XML text node child of e.
    //! If attrName if given evaluate it "fully" if fullEval is true or "partially".
    //! The result of a "partially" evaluation is returned as a octave string even so it is not really a string.
    octave_value eval(const TiXmlElement *e, const std::string &attrName=std::string(), bool fullEval=true);

    /*! Cast the octave value value to type <tt>T</tt>.
     * Possible combinations of allowed octave value types and template types <tt>T</tt> are listed in the
     * following table. If a combination is not allowed a exception is thrown, except for casts which are marked
     * as not type save in the table.
     * If a c-pointer is returned this c-pointer is only guaranteed to be valid for the lifetime of the octave object
     * \p value being passed as argument.
     * <table>
     * <tr><td></td><th colspan="7"><tt>value</tt> is of Octave Type ...</th></tr>
     * <tr><th>Template Type <tt>T</tt> equals ...</th><th>scalar</th><th>(row) vector</th><th>matrix</th><th>string</th><th><i>SWIG</i> <tt>CasADi::SXFunction</tt></th><th><i>SWIG</i> <tt>CasADi::SXMatrix</tt></th><th><i>SWIG</i> <tt>XYZ</tt></th></tr>
     * <tr><th><tt>double</tt></th><td>O</td><td></td><td></td><td></td><td></td><td></td><td></td></tr>
     * <tr><th><tt>vector&lt;double&gt;</tt></th><td>O</td><td>O</td><td></td><td></td><td></td><td></td><td></td></tr>
     * <tr><th><tt>vector&lt;vector&lt;double&gt;&nbsp;&gt;</tt></th><td>O</td><td>O</td><td>O</td><td></td><td></td><td></td><td></td></tr>
     * <tr><th><tt>string</tt></th><td>O (returns e.g. "5")</td><td>O (returns e.g. "[3;5]")</td><td>O (returns e.g. "[1,2;3,4]")</td><td>O (returns e.g. "'foo'"</td><td></td><td></td><td></td></tr>
     * <tr><th><tt>auto_ptr&lt;TiXmlElement&gt;</tt></th><td></td><td></td><td></td><td></td><td>O (in MBXMLUtilsTinyXML representation)</td><td></td><td></td></tr>
     * <tr><th><tt>CasADi::SXMatrix</tt></th><td>O</td><td>O</td><td>O</td><td></td><td></td><td>O</td><td></td></tr>
     * <tr><th><tt>CasADi::SXMatrix*</tt></th><td></td><td></td><td></td><td></td><td></td><td>O</td><td></td></tr>
     * <tr><th><tt>CasADi::SXFunction</tt></th><td></td><td></td><td></td><td></td><td>O</td><td></td><td></td></tr>
     * <tr><th><tt>CasADi::SXFunction*</tt></th><td></td><td></td><td></td><td></td><td>O</td><td></td><td></td></tr>
     * <tr><th><tt>XYZ</tt></th><td></td><td></td><td></td><td></td><td></td><td></td><td>O (not type save)</td></tr>
     * <tr><th><tt>XYZ*</tt></th><td></td><td></td><td></td><td></td><td></td><td></td><td>O (not type save)</td></tr>
     * </table>
     */
    template<typename T>
    T cast(const octave_value &value);

    //! get the type of value
    ValueType getType(const octave_value &value);

  protected:
    //! Push the current parameters to a internal stack.
    void pushParams();

    //! Overwrite the current parameter with the top level set from the internal stack.
    void popParams();

    //! create octave value of CasADi type name. Created using the default ctor.
    octave_value createCasADi(const std::string &name);

    // clone the octave value of type SWIG to type T
    template<typename T>
    T cloneSwigAs(const octave_value &value);

    octave_value stringToOctValue(const std::string &str, const TiXmlElement *e) const;

    // check if value is of a swig object of type T
    template<typename T>
    void isSwig(const octave_value &value);

    // map of the current parameters
    std::unordered_map<std::string, octave_value> currentParam;

    // stack of parameters
    std::stack<std::unordered_map<std::string, octave_value> > paramStack;

    static int initCount;

    static std::map<std::string, std::string> units;

    octave_value casadiOctValue;
};

// Helper class which convert a void* to T* or T.
template<typename T>
struct Ptr {
  static T cast(void *ptr) { return *static_cast<T*>(ptr); }
};
template<typename T>
struct Ptr<T*> {
  static T* cast(void *ptr) { return static_cast<T*>(ptr); }
};

// default: not cloneable
template<typename T>
T OctEval::cloneSwigAs(const octave_value &value) {
  throw std::runtime_error("Cannot clone this octave_value of type SWIG.");
}

template<> CasADi::SXMatrix OctEval::cloneSwigAs<CasADi::SXMatrix>(const octave_value &value);

// default: do no type check to allow all conversion is a type unsafe way
template<typename T>
void OctEval::isSwig(const octave_value &value) {
}

template<> void OctEval::isSwig<CasADi::SXMatrix>(const octave_value &value);
template<> void OctEval::isSwig<CasADi::SXMatrix*>(const octave_value &value);
template<> void OctEval::isSwig<CasADi::SXFunction>(const octave_value &value);
template<> void OctEval::isSwig<CasADi::SXFunction*>(const octave_value &value);

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
  // try to clone it if value is not a SWIG object
  if(error_state!=0) {
    error_state=0;
    return cloneSwigAs<T>(value);
  }
  // type check
  isSwig<T>(value);
  void *swigPtr=reinterpret_cast<void*>(swigThis.uint64_scalar_value().value());
  // convert the void pointer to the correct casadi type
  return Ptr<T>::cast(swigPtr);
}

template<> std::string OctEval::cast<std::string>(const octave_value &value);
template<> double OctEval::cast<double>(const octave_value &value);
template<> std::vector<double> OctEval::cast<std::vector<double> >(const octave_value &value);
template<> std::vector<std::vector<double> > OctEval::cast<std::vector<std::vector<double> > >(const octave_value &value);
template<> std::auto_ptr<MBXMLUtils::TiXmlElement> OctEval::cast<std::auto_ptr<MBXMLUtils::TiXmlElement> >(const octave_value &value);

} // end namespace MBXMLUtils

#endif
