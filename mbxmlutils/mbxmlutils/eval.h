#ifndef _MBXMLUTILS_EVAL_H_
#define _MBXMLUTILS_EVAL_H_

#include <fmatvec/atom.h>
#include <boost/filesystem.hpp>
#include <xercesc/util/XercesDefs.hpp>
#include <casadi/core/function/sx_function.hpp>
#ifdef HAVE_UNORDERED_MAP
#  include <unordered_map>
#else
#  include <map>
#  define unordered_map map
#endif

namespace XERCES_CPP_NAMESPACE {
  class DOMElement;
  class DOMAttr;
  class DOMDocument;
}

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

class Eval;

//! Create a new parameter level for a evaluator which is automatically resetted if the scope of this object is left.
class NewParamLevel {
  public:
    //! Create a new parameter level in the evaluator oe_
    NewParamLevel(Eval &oe_, bool newLevel_=true);
    //! Reset to the previous parameter level
    ~NewParamLevel();
  protected:
    Eval &oe;
    bool newLevel;
};

/*! Expression evaluator and converter. */
class Eval : virtual public fmatvec::Atom {
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

  protected:
    //! Constructor.
    Eval(std::vector<boost::filesystem::path> *dependencies_=NULL);
  public:
    //! Destructor.
    ~Eval();

    //! Create a evaluator.
    static boost::shared_ptr<Eval> createEvaluator(const std::string &evalName, std::vector<boost::filesystem::path> *dependencies_=NULL);

    //! Get the type of this evaluator
    virtual std::string getEvaluatorName()=0;

    //! Add a value to the current parameters.
    virtual void addParam(const std::string &paramName, const boost::shared_ptr<void>& value);
    //! Add all parameters from XML element e.
    //! The parameters are added from top to bottom as they appear in the XML element e.
    //! Parameters may depend on parameters already added.
    virtual void addParamSet(const xercesc::DOMElement *e);

    //! Add dir to the evauator search path.
    //! A relative path in dir is expanded to an absolute path using the current directory.
    virtual void addPath(const boost::filesystem::path &dir, const xercesc::DOMElement *e)=0;
    
    //! Evaluate the XML element e using the current parameters returning the resulting value.
    //! The type of evaluation depends on the type of e.
    virtual boost::shared_ptr<void> eval(const xercesc::DOMElement *e)=0;

    //! Evaluate the XML attribute a using the current parameters returning the resulting value.
    //! The type of evaluation depends on the type of a.
    //! The result of a "partially" evaluation is returned as a string even so it is not really a string.
    virtual boost::shared_ptr<void> eval(const xercesc::DOMAttr *a, const xercesc::DOMElement *pe=NULL)=0;

    /*! Cast the value value to type <tt>T</tt>.
     * Possible combinations of allowed value types and template types <tt>T</tt> are listed in the
     * following table. If a combination is not allowed a exception is thrown, except for casts which are marked
     * as not type save in the table.
     * If a c-pointer is returned this c-pointer is only guaranteed to be valid for the lifetime of the object
     * \p value being passed as argument.
     * If a DOMElement* is returned \p doc must be given and ownes the memory of the returned DOM tree. For other
     * return types this function must be called with only one argument cast(const boost::shared_ptr<void> &value);
     * <table>
     *   <tr>
     *     <th></th>
     *     <th colspan="6"><tt>value</tt> is of Type ...</th>
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
    T cast(const boost::shared_ptr<void> &value, xercesc::DOMDocument *doc);

    //! see cast(const boost::shared_ptr<void> &value, shared_ptr<DOMDocument> &doc)
    template<typename T>
    T cast(const boost::shared_ptr<void> &value);

    //! get the type of value
    virtual ValueType getType(const boost::shared_ptr<void> &value)=0;

    //! evaluate str and return result as an variable, this can be used to evaluate outside of XML.
    //! If e is given it is used as location information in case of errors.
    //! If fullEval is false the "partially" evaluation is returned as a string even so it is not really a string.
    virtual boost::shared_ptr<void> stringToValue(const std::string &str, const xercesc::DOMElement *e=NULL, bool fullEval=true)=0;

    //! create a value of the given type
    template<class T>
    boost::shared_ptr<void> create(const T& v);

    //! return a list of all required files of the evaluator (excluding dependent files of libraries)
    virtual std::map<boost::filesystem::path, std::pair<boost::filesystem::path, bool> >& requiredFiles()=0;

  protected:
    //! Push the current parameters to a internal stack.
    void pushParams();

    //! Overwrite the current parameter with the top level set from the internal stack.
    void popParams();

    //! Push the current path to a internal stack.
    void pushPath();

    //! Overwrite the current path with the top level path from the internal stack.
    void popPath();

    std::vector<boost::filesystem::path> *dependencies;

    // map of the current parameters
    std::unordered_map<std::string, boost::shared_ptr<void> > currentParam;
    // stack of parameters
    std::stack<std::unordered_map<std::string, boost::shared_ptr<void> > > paramStack;
    // stack of path
    std::stack<std::string> pathStack;

  private:
    virtual void* castToSwig(const boost::shared_ptr<void> &value)=0;

    // virtual spezialization of cast(const boost::shared_ptr<void> &value)
    virtual int                               cast_int                 (const boost::shared_ptr<void> &value)=0;
    virtual double                            cast_double              (const boost::shared_ptr<void> &value)=0;
    virtual std::vector<double>               cast_vector_double       (const boost::shared_ptr<void> &value)=0;
    virtual std::vector<std::vector<double> > cast_vector_vector_double(const boost::shared_ptr<void> &value)=0;
    virtual std::string                       cast_string              (const boost::shared_ptr<void> &value)=0;
    virtual casadi::SXFunction                cast_SXFunction          (const boost::shared_ptr<void> &value)=0;
    virtual casadi::SXFunction*               cast_SXFunction_p        (const boost::shared_ptr<void> &value)=0;
    virtual casadi::SX                        cast_SX                  (const boost::shared_ptr<void> &value)=0;
    virtual casadi::SX*                       cast_SX_p                (const boost::shared_ptr<void> &value)=0;
    virtual casadi::DMatrix                   cast_DMatrix             (const boost::shared_ptr<void> &value)=0;
    virtual casadi::DMatrix*                  cast_DMatrix_p           (const boost::shared_ptr<void> &value)=0;
    // virtual spezialization of cast(const boost::shared_ptr<void> &value, xercesc::DOMDocument *doc)
    virtual xercesc::DOMElement*              cast_DOMElement_p        (const boost::shared_ptr<void> &value, xercesc::DOMDocument *doc)=0;

    // virtual spezialization of create(...)
    virtual boost::shared_ptr<void> create_double              (const double& v)=0;
    virtual boost::shared_ptr<void> create_vector_double       (const std::vector<double>& v)=0;
    virtual boost::shared_ptr<void> create_vector_vector_double(const std::vector<std::vector<double> >& v)=0;
    virtual boost::shared_ptr<void> create_string              (const std::string& v)=0;
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

template<typename T>
T Eval::cast(const boost::shared_ptr<void> &value) {
  // do not allow T == DOMElement* here ...
  BOOST_STATIC_ASSERT_MSG((boost::is_same<T, xercesc::DOMElement*>::type), 
    "Calling Eval::cast<DOMElement*>(const boost::shared_ptr<void>&) is not allowed "
    "use Eval::cast<DOMElement*>(const boost::shared_ptr<void>&, DOMDocument*)"
  );
  // ... but treat all other type as swig types ...
  return Ptr<T>::cast(castToSwig(value));
}
// ... but prevere these specializations
template<> std::string Eval::cast<std::string>(const boost::shared_ptr<void> &value);
template<> int Eval::cast<int>(const boost::shared_ptr<void> &value);
template<> double Eval::cast<double>(const boost::shared_ptr<void> &value);
template<> std::vector<double> Eval::cast<std::vector<double> >(const boost::shared_ptr<void> &value);
template<> std::vector<std::vector<double> > Eval::cast<std::vector<std::vector<double> > >(const boost::shared_ptr<void> &value);
template<> casadi::SX Eval::cast<casadi::SX>(const boost::shared_ptr<void> &value);
template<> casadi::SX* Eval::cast<casadi::SX*>(const boost::shared_ptr<void> &value);
template<> casadi::SXFunction Eval::cast<casadi::SXFunction>(const boost::shared_ptr<void> &value);
template<> casadi::SXFunction* Eval::cast<casadi::SXFunction*>(const boost::shared_ptr<void> &value);
template<> casadi::DMatrix Eval::cast<casadi::DMatrix>(const boost::shared_ptr<void> &value);
template<> casadi::DMatrix* Eval::cast<casadi::DMatrix*>(const boost::shared_ptr<void> &value);

template<typename T>
T Eval::cast(const boost::shared_ptr<void> &value, xercesc::DOMDocument *doc) {
  // only allow T == DOMElement* here ...
  BOOST_STATIC_ASSERT_MSG(!(boost::is_same<T, xercesc::DOMElement*>::type), 
    "Calling Eval::cast<T>(const boost::shared_ptr<void>&, DOMDocument*) is only allowed for T=DOMElement*"
  );
}
// ... but use these spezializations
template<> xercesc::DOMElement* Eval::cast<xercesc::DOMElement*>(const boost::shared_ptr<void> &value, xercesc::DOMDocument *doc);

// spezializations for create
template<> boost::shared_ptr<void> Eval::create<double>                            (const double& v);
template<> boost::shared_ptr<void> Eval::create<std::vector<double> >              (const std::vector<double>& v);
template<> boost::shared_ptr<void> Eval::create<std::vector<std::vector<double> > >(const std::vector<std::vector<double> >& v);
template<> boost::shared_ptr<void> Eval::create<std::string>                       (const std::string& v);

} // end namespace MBXMLUtils

#endif
