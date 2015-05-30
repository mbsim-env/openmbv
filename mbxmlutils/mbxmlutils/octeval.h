#ifndef _MBXMLUTILS_OCTEVAL_H_
#define _MBXMLUTILS_OCTEVAL_H_

#include <fmatvec/atom.h>
#include <string>
#include <stack>
#ifdef HAVE_UNORDERED_MAP
#  include <unordered_map>
#else
#  include <map>
#  define unordered_map map
#endif

// Include octave/config.h first. This is normally not allowed since config.h should only
// be included in .cc files but is required by octave.
// To avoid macro redefined warnings/errors we undefine it before
// including octave/oct.h. Note that we can not restore the values. So you have to reinclude
// your config.h after this file to get the original values.
// undef macros
#undef PACKAGE
#undef PACKAGE_BUGREPORT
#undef PACKAGE_NAME
#undef PACKAGE_STRING
#undef PACKAGE_TARNAME
#undef PACKAGE_URL
#undef PACKAGE_VERSION
#undef VERSION
// include
#include <octave/config.h>
// undef macros
#undef PACKAGE
#undef PACKAGE_BUGREPORT
#undef PACKAGE_NAME
#undef PACKAGE_STRING
#undef PACKAGE_TARNAME
#undef PACKAGE_URL
#undef PACKAGE_VERSION
#undef VERSION
// We cannot reset the macros so you have to reinclude your config.h after this file
// to get the original values

#include <octave/symtab.h>
#include "mbxmlutilshelper/casadiXML.h"
#include "mbxmlutilshelper/dom.h"
#include <octave/parse.h>
#include <boost/filesystem.hpp>
#include <boost/static_assert.hpp> 
#include <boost/scoped_ptr.hpp>
#include <xercesc/util/XercesDefs.hpp>
#include <xercesc/dom/DOMDocument.hpp>

namespace XERCES_CPP_NAMESPACE { class DOMElement; }

namespace MBXMLUtils {

extern bool deactivateBlock;
// A class to block/unblock stderr or stdout. Block in called in the ctor, unblock in the dtor
template<int T>
class Block {
  public:
    Block(std::ostream &str_, std::streambuf *buf=NULL) : str(str_) {
      if(deactivateBlock) return;
      if(disableCount==0)
        orgcxxx=str.rdbuf(buf);
      disableCount++;
    }
    ~Block() {
      if(deactivateBlock) return;
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
#define BLOCK_STDOUT(name) Block<1> name(std::cout)
#define BLOCK_STDERR(name) Block<2> name(std::cerr)
#define REDIR_STDOUT(name, buf) Block<1> name(std::cout, buf)
#define REDIR_STDERR(name, buf) Block<2> name(std::cerr, buf)

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
class OctEval : virtual public fmatvec::Atom {
  public:
    friend class NewParamLevel;

    //! Known type for the "ret" variable
    enum ValueType {
      ScalarType,
      VectorType,
      MatrixType,
      StringType,
      SXType,
      DMatrixType,
      SXFunctionType
    };

    //! Constructor.
    OctEval(std::vector<boost::filesystem::path> *dependencies_=NULL);
    //! Destructor.
    ~OctEval();

    //! Add a octave value to the current parameters.
    void addParam(const std::string &paramName, const octave_value& value);
    //! Add all parameters from XML element e.
    //! The parameters are added from top to bottom as they appear in the XML element e.
    //! Parameters may depend on parameters already added.
    void addParamSet(const xercesc::DOMElement *e);

    //! Add dir to octave search path
    //! A relative path in dir is expanded to an absolute path using the current directory.
    void addPath(const boost::filesystem::path &dir, const xercesc::DOMElement *e);
    
    //! Evaluate the XML element e using the current parameters returning the resulting octave value.
    //! The type of evaluation depends on the type of e.
    octave_value eval(const xercesc::DOMElement *e);

    //! Evaluate the XML attribute a using the current parameters returning the resulting octave value.
    //! The type of evaluation depends on the type of a.
    //! The result of a "partially" evaluation is returned as a octave string even so it is not really a string.
    octave_value eval(const xercesc::DOMAttr *a, const xercesc::DOMElement *pe=NULL);

    /*! Cast the octave value value to type <tt>T</tt>.
     * Possible combinations of allowed octave value types and template types <tt>T</tt> are listed in the
     * following table. If a combination is not allowed a exception is thrown, except for casts which are marked
     * as not type save in the table.
     * If a c-pointer is returned this c-pointer is only guaranteed to be valid for the lifetime of the octave object
     * \p value being passed as argument.
     * If a DOMElement* is returned \p doc must be given and ownes the memory of the returned DOM tree. For other
     * return types this function must be called with only one argument cast(const octave_value &value);
     * <table>
     *   <tr>
     *     <th></th>
     *     <th colspan="6"><tt>value</tt> is of Octave Type ...</th>
     *   </tr>
     *   <tr>
     *     <th>Template Type <tt>T</tt> equals ...</th>
     *     <th>real</th>
     *     <th>string</th>
     *     <th><i>SWIG</i> <tt>casadi::SXFunction</tt></th>
     *     <th><i>SWIG</i> <tt>casadi::SX</tt></th>
     *     <th><i>SWIG</i> <tt>casadi::DMatrix</tt></th>
     *     <th><i>SWIG</i> <tt>XYZ</tt></th>
     *   </tr>
     *
     *   <tr>
     *     <!--CAST TO-->    <th><tt>int</tt></th>
     *     <!--real-->       <td>only if 1 x 1 and a integral number</td>
     *     <!--string-->     <td></td>
     *     <!--SXFunction--> <td></td>
     *     <!--SX-->         <td>only if 1 x 1 and constant and a integral number</td>
     *     <!--DMatrix-->    <td>only if 1 x 1 and a integral number</td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>double</tt></th>
     *     <!--real-->       <td>only if 1 x 1</td>
     *     <!--string-->     <td></td>
     *     <!--SXFunction--> <td></td>
     *     <!--SX-->         <td>only if 1 x 1 and constant</td>
     *     <!--DMatrix-->    <td>only if 1 x 1</td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>vector&lt;double&gt;</tt></th>
     *     <!--real-->       <td>only if n x 1</td>
     *     <!--string-->     <td></td>
     *     <!--SXFunction--> <td></td>
     *     <!--SX-->         <td>only if n x 1 and constant</td>
     *     <!--DMatrix-->    <td>only if n x 1</td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>vector&lt;vector&lt;double&gt;&nbsp;&gt;</tt></th>
     *     <!--real-->       <td>X</td>
     *     <!--string-->     <td></td>
     *     <!--SXFunction--> <td></td>
     *     <!--SX-->         <td>only if constant</td>
     *     <!--DMatrix-->    <td>X</td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>string</tt></th>
     *     <!--real-->       <td>returns e.g. "5" or "[1,3;5,4]"</td>
     *     <!--string-->     <td>returns e.g. "'foo'"</td>
     *     <!--SXFunction--> <td></td>
     *     <!--SX-->         <td>only if constant; returns e.g. "5" or "[1,3;5,4]"</td>
     *     <!--DMatrix-->    <td>returns e.g. "5" or "[1,3;5,4]"</td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>DOMElement*</tt></th>
     *     <!--real-->       <td></td>
     *     <!--string-->     <td></td>
     *     <!--SXFunction--> <td>X</td>
     *     <!--SX-->         <td></td>
     *     <!--DMatrix-->    <td></td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>casadi::SXFunction</tt></th>
     *     <!--real-->       <td></td>
     *     <!--string-->     <td></td>
     *     <!--SXFunction--> <td>X</td>
     *     <!--SX-->         <td></td>
     *     <!--DMatrix-->    <td></td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>casadi::SXFunction*</tt></th>
     *     <!--real-->       <td></td>
     *     <!--string-->     <td></td>
     *     <!--SXFunction--> <td>X</td>
     *     <!--SX-->         <td></td>
     *     <!--DMatrix-->    <td></td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>casadi::SX</tt></th>
     *     <!--real-->       <td>X</td>
     *     <!--string-->     <td></td>
     *     <!--SXFunction--> <td></td>
     *     <!--SX-->         <td>X</td>
     *     <!--DMatrix-->    <td>X</td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>casadi::SX*</tt></th>
     *     <!--real-->       <td></td>
     *     <!--string-->     <td></td>
     *     <!--SXFunction--> <td></td>
     *     <!--SX-->         <td>X</td>
     *     <!--DMatrix-->    <td></td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>casadi::DMatrix</tt></th>
     *     <!--real-->       <td>X</td>
     *     <!--string-->     <td></td>
     *     <!--SXFunction--> <td></td>
     *     <!--SX-->         <td>X</td>
     *     <!--DMatrix-->    <td>X</td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>casadi::DMatrix*</tt></th>
     *     <!--real-->       <td></td>
     *     <!--string-->     <td></td>
     *     <!--SXFunction--> <td></td>
     *     <!--SX-->         <td></td>
     *     <!--DMatrix-->    <td>X</td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>XYZ</tt></th>
     *     <!--real-->       <td></td>
     *     <!--string-->     <td></td>
     *     <!--SXFunction--> <td></td>
     *     <!--SX-->l        <td></td>
     *     <!--DMatrix-->    <td></td>
     *     <!--XYZ-->        <td>not type save</td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>XYZ*</tt></th>
     *     <!--real-->       <td></td>
     *     <!--string-->     <td></td>
     *     <!--SXFunction--> <td></td>
     *     <!--SX-->         <td></td>
     *     <!--DMatrix-->    <td></td>
     *     <!--XYZ-->        <td>not type save</td>
     *   </tr>
     * </table>
     */
    template<typename T>
    static T cast(const octave_value &value, xercesc::DOMDocument *doc);

    //! see cast(const octave_value &value, shared_ptr<DOMDocument> &doc)
    template<typename T>
    static T cast(const octave_value &value);

    //! get the type of value
    static ValueType getType(const octave_value &value);

    //! evaluate str and return result as an octave variable, this can be used to evaluate outside of XML.
    //! If e is given it is used as location information in case of errors.
    //! If fullEval is false the "partially" evaluation is returned as a octave string even so it is not really a string.
    octave_value stringToOctValue(const std::string &str, const xercesc::DOMElement *e=NULL, bool fullEval=true) const;

  protected:

    //! This function deinitialized octave. It is used in the dtor and before exceptions in the ctor are thrown
    // (in the later case the dtor is not called but octave must be uninitialized before exit)
    void deinitOctave();

    //! evaluate str fully and return result as an octave variable
    octave_value fullStringToOctValue(const std::string &str, const xercesc::DOMElement *e=NULL) const;

    //! evaluate str partially and return result as an std::string
    std::string partialStringToOctValue(const std::string &str, const xercesc::DOMElement *e) const;

    //! Push the current parameters to a internal stack.
    void pushParams();

    //! Overwrite the current parameter with the top level set from the internal stack.
    void popParams();

    //! Push the current path to a internal stack.
    void pushPath();

    //! Overwrite the current path with the top level path from the internal stack.
    void popPath();

    //! cast value to the corresponding swig object of type T, without ANY type check.
    template<typename T>
    static T castToSwig(const octave_value &value);

    std::vector<boost::filesystem::path> *dependencies;

    //! create octave value of CasADi type name. Created using the default ctor.
    static octave_value createCasADi(const std::string &name);

    // map of the current parameters
    std::unordered_map<std::string, octave_value> currentParam;
    // initial path
    static std::string initialPath;
    static std::string pathSep;

    // stack of parameters
    std::stack<std::unordered_map<std::string, octave_value> > paramStack;
    // stack of path
    std::stack<std::string> pathStack;

    static int initCount;

    static std::map<std::string, std::string> units;

    static boost::scoped_ptr<octave_value> casadiOctValue;

    static octave_value_list fevalThrow(octave_function *func, const octave_value_list &arg, int n=0,
                                        const std::string &msg=std::string());

    octave_value handleUnit(const xercesc::DOMElement *e, const octave_value &ret);
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

// cast octave value to swig object ptr or swig object copy
template<typename T>
T OctEval::castToSwig(const octave_value &value) {
  // get the casadi pointer: octave returns a 64bit integer which represents the pointer
  static octave_function *swig_this=symbol_table::find_function("swig_this").function_value(); // get ones a pointer to swig_this for performance reasons
  octave_value swigThis;
  {
    BLOCK_STDERR(blockstderr);
    swigThis=feval(swig_this, value, 1)(0);
  }
  if(error_state!=0)
    throw std::runtime_error("Internal error: Not a swig object");
  void *swigPtr=reinterpret_cast<void*>(swigThis.uint64_scalar_value().value());
  // convert the void pointer to the correct casadi type
  return Ptr<T>::cast(swigPtr);
}

template<typename T>
T OctEval::cast(const octave_value &value) {
  // do not allow T == DOMElement* here ...
  BOOST_STATIC_ASSERT_MSG((boost::is_same<T, xercesc::DOMElement*>::type), 
    "Calling OctEval::cast<DOMElement*>(const octave_value&) is not allowed "
    "use OctEval::cast<DOMElement*>(const octave_value&, DOMDocument*)"
  );
  // ... but treat all other type as octave swig types ...
  return castToSwig<T>(value);
}
// ... but prevere these specializations
template<> std::string OctEval::cast<std::string>(const octave_value &value);
template<> long OctEval::cast<long>(const octave_value &value);
template<> double OctEval::cast<double>(const octave_value &value);
template<> std::vector<double> OctEval::cast<std::vector<double> >(const octave_value &value);
template<> std::vector<std::vector<double> > OctEval::cast<std::vector<std::vector<double> > >(const octave_value &value);
template<> casadi::SX OctEval::cast<casadi::SX>(const octave_value &value);
template<> casadi::SX* OctEval::cast<casadi::SX*>(const octave_value &value);
template<> casadi::SXFunction OctEval::cast<casadi::SXFunction>(const octave_value &value);
template<> casadi::SXFunction* OctEval::cast<casadi::SXFunction*>(const octave_value &value);
template<> casadi::DMatrix OctEval::cast<casadi::DMatrix>(const octave_value &value);
template<> casadi::DMatrix* OctEval::cast<casadi::DMatrix*>(const octave_value &value);

template<typename T>
T OctEval::cast(const octave_value &value, xercesc::DOMDocument *doc) {
  // only allow T == DOMElement* here ...
  BOOST_STATIC_ASSERT_MSG(!(boost::is_same<T, xercesc::DOMElement*>::type), 
    "Calling OctEval::cast<T>(const octave_value&, DOMDocument*) is only allowed for T=DOMElement*"
  );
}
// ... but use these spezializations
template<> xercesc::DOMElement* OctEval::cast<xercesc::DOMElement*>(const octave_value &value, xercesc::DOMDocument *doc);

} // end namespace MBXMLUtils

#endif
