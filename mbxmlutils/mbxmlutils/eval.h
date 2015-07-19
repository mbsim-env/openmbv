#ifndef _MBXMLUTILS_EVAL_H_
#define _MBXMLUTILS_EVAL_H_

#include <fmatvec/atom.h>
#include <boost/filesystem.hpp>
#include <boost/function.hpp>
#include <xercesc/util/XercesDefs.hpp>
#include <mbxmlutilshelper/dom.h>
#include <casadi/core/function/sx_function.hpp>
#ifdef HAVE_UNORDERED_MAP
#  include <unordered_map>
#else
#  include <map>
#  define unordered_map map
#endif

#define XMLUTILS_EVAL_CONCAT1(X, Y) X##Y
#define XMLUTILS_EVAL_CONCAT(X, Y) XMLUTILS_EVAL_CONCAT1(X, Y)
#define XMLUTILS_EVAL_APPENDLINE(X) XMLUTILS_EVAL_CONCAT(X, __LINE__)

/** Use this macro to register a new evaluator */
#define XMLUTILS_EVAL_REGISTER(T) \
  namespace { \
    struct Reg { \
      Reg() { Eval::registerEvaluator<T>(); } \
    } XMLUTILS_EVAL_APPENDLINE(regDummy); \
  }

namespace XERCES_CPP_NAMESPACE {
  class DOMElement;
  class DOMAttr;
  class DOMDocument;
}

namespace MBXMLUtils {

//! A dummy object representing a value as string in the syntax of the Eval.
class CodeString : public std::string {
  public:
    CodeString(const std::string &str) : std::string(str) {}
};

// Store the current directory in the ctor an restore in the dtor
class PreserveCurrentDir {
  public:
    PreserveCurrentDir() {
      dir=boost::filesystem::current_path();
    }
    ~PreserveCurrentDir() {
      boost::filesystem::current_path(dir);
    }
  private:
    boost::filesystem::path dir;
};

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

template<class T> struct SwigType { static std::string name; };

/*! Expression evaluator and converter. */
class Eval : virtual public fmatvec::Atom {
  public:
    friend class NewParamLevel;

    //! Known types for a variable
    //! See also valueIsOfType.
    enum ValueType {
      ScalarType,
      VectorType,
      MatrixType,
      StringType,
      SXFunctionType
    };

  protected:
    //! Constructor.
    Eval(std::vector<boost::filesystem::path> *dependencies_);
  public:
    //! Destructor.
    ~Eval();

    //! Create a evaluator.
    static boost::shared_ptr<Eval> createEvaluator(const std::string &evalName, std::vector<boost::filesystem::path> *dependencies_=NULL);

    // Register a new evaluator.
    template<class E>
    static void registerEvaluator() {
      getEvaluators()[E::getNameStatic()]=&newEvaluator<E>;
    }

    //! Get the type of this evaluator
    virtual std::string getName() const=0;

    //! Add a value to the current parameters.
    void addParam(const std::string &paramName, const boost::shared_ptr<void>& value);
    //! Add all parameters from XML element e.
    //! The parameters are added from top to bottom as they appear in the XML element e.
    //! Parameters may depend on parameters already added.
    void addParamSet(const xercesc::DOMElement *e);

    //! Import evaluator statements. This routine highly depends on the evaluator.
    //! See the spezialized evaluators documentation for details.
    virtual void addImport(const std::string &code, const xercesc::DOMElement *e, bool deprecated=false)=0;// MISSING: fullEval is just to support a deprected feature
    
    //! Evaluate the XML element e using the current parameters returning the resulting value.
    //! The type of evaluation depends on the type of e.
    boost::shared_ptr<void> eval(const xercesc::DOMElement *e);

    //! Evaluate the XML attribute a using the current parameters returning the resulting value.
    //! The type of evaluation depends on the type of a.
    //! The result of a "partially" evaluation is returned as a string even so it is not really a string.
    boost::shared_ptr<void> eval(const xercesc::DOMAttr *a, const xercesc::DOMElement *pe=NULL);

    /*! Cast the value value to type <tt>T</tt>.
     * Possible combinations of allowed value types and template types <tt>T</tt> are listed in the
     * following table. If a combination is not allowed a exception is thrown.
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
     *     <th>real scalar</th>
     *     <th>real vector</th>
     *     <th>real matrix</th>
     *     <th>string</th>
     *     <th><i>SWIG</i> <tt>casadi::SXFunction</tt></th>
     *     <th><i>SWIG</i> <tt>XYZ</tt></th>
     *   </tr>
     *
     *   <tr>
     *     <!--CAST TO-->    <th><tt>int</tt></th>
     *     <!--scalar-->     <td>only if its integral number</td>
     *     <!--vector-->     <td></td>
     *     <!--matrix-->     <td></td>
     *     <!--string-->     <td></td>
     *     <!--SXFunction--> <td></td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>double</tt></th>
     *     <!--scalar-->     <td>X</td>
     *     <!--vector-->     <td></td>
     *     <!--matrix-->     <td></td>
     *     <!--string-->     <td></td>
     *     <!--SXFunction--> <td></td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>vector&lt;double&gt;</tt></th>
     *     <!--scalar-->     <td>X</td>
     *     <!--vector-->     <td>X</td>
     *     <!--matrix-->     <td></td>
     *     <!--string-->     <td></td>
     *     <!--SXFunction--> <td></td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>vector&lt;vector&lt;double&gt;&nbsp;&gt;</tt></th>
     *     <!--scalar-->     <td>X</td>
     *     <!--vector-->     <td>X</td>
     *     <!--matrix-->     <td>X</td>
     *     <!--string-->     <td></td>
     *     <!--SXFunction--> <td></td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>string</tt></th>
     *     <!--scalar-->     <td></td>
     *     <!--vector-->     <td></td>
     *     <!--matrix-->     <td></td>
     *     <!--string-->     <td>X</td>
     *     <!--SXFunction--> <td></td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>CodeString</tt></th>
     *     <!--scalar-->     <td>returns e.g. <code>5</code></td>
     *     <!--vector-->     <td>returns e.g. <code>[3;7]</code></td>
     *     <!--matrix-->     <td>returns e.g. <code>[1,3;5,4]</code></td>
     *     <!--string-->     <td>returns e.g. <code>'foo'</code></td>
     *     <!--SXFunction--> <td></td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>DOMElement*</tt></th>
     *     <!--scalar-->     <td></td>
     *     <!--vector-->     <td></td>
     *     <!--matrix-->     <td></td>
     *     <!--string-->     <td></td>
     *     <!--SXFunction--> <td>X</td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>casadi::SXFunction*</tt></th>
     *     <!--scalar-->     <td></td>
     *     <!--vector-->     <td></td>
     *     <!--matrix-->     <td></td>
     *     <!--string-->     <td></td>
     *     <!--SXFunction--> <td>X</td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>XYZ*</tt></th>
     *     <!--scalar-->     <td></td>
     *     <!--vector-->     <td></td>
     *     <!--matrix-->     <td></td>
     *     <!--string-->     <td></td>
     *     <!--SXFunction--> <td></td>
     *     <!--XYZ-->        <td>*</td>
     *   </tr>
     * </table>
     * For arbitary evaluator values of type SWIG wrapper object (see * in table) a instantiation of
     * <code>template<> string SwigType<XYZ*>::name("<swig_type_name_of_XYZ>");</code>
     * is required. Eval instantiates this for casadi::SX and casadi::SXFunction.
     */
    template<typename T>
    T cast(const boost::shared_ptr<void> &value, xercesc::DOMDocument *doc) const;

    //! see cast(const boost::shared_ptr<void> &value, shared_ptr<DOMDocument> &doc)
    template<typename T>
    T cast(const boost::shared_ptr<void> &value) const;

    //! check whether value is of type type.
    //! Note that true is only returned if the value is really of type type. If value can be casted
    //! to type type but is not of this type then false is returned.
    //! Note that their are evaluators (e.g. octave) which cannot distinguish between e.g. a scalar,
    //! a vector of size 1 or a matrix of size 1x1. Hence, these evaluators will return true for ScalarType
    //! in all these three cases and analog for VectorType.
    virtual bool valueIsOfType(const boost::shared_ptr<void> &value, ValueType type) const=0;

    //! evaluate str and return result as an variable, this can be used to evaluate outside of XML.
    //! If e is given it is used as location information in case of errors.
    //! If fullEval is false the "partially" evaluation is returned as a string even so it is not really a string.
    boost::shared_ptr<void> stringToValue(const std::string &str, const xercesc::DOMElement *e=NULL, bool fullEval=true) const;

    //! create a value of the given type
    template<class T>
    boost::shared_ptr<void> create(const T& v) const;

    //! return a list of all required files of the evaluator (excluding dependent files of libraries)
    virtual std::map<boost::filesystem::path, std::pair<boost::filesystem::path, bool> >& requiredFiles() const=0;

    //! Return true if the evaluator used one based indexes or false if zero based indexes are used.
    virtual bool useOneBasedIndexes()=0;

  protected:
    //! Push the current context to a internal stack.
    void pushContext();

    //! Overwrite the current context with the top level context from the internal stack.
    void popContext();

    std::vector<boost::filesystem::path> *dependencies;

    // map of the current parameters
    std::unordered_map<std::string, boost::shared_ptr<void> > currentParam;
    // stack of parameters
    std::stack<std::unordered_map<std::string, boost::shared_ptr<void> > > paramStack;

    // current imports
    boost::shared_ptr<void> currentImport;
    // stack of imports
    std::stack<boost::shared_ptr<void> > importStack;

    template<typename T>
    boost::shared_ptr<void> createSwig() const;

    //! create a SWIG object of name typeName.
    virtual boost::shared_ptr<void> createSwigByTypeName(const std::string &typeName) const=0;

    /*! Return the value of a call to name using the arguments args.
     * The following functions must be implemented by the evaluator:
     *   - rotateAboutX(alpha): returns a 3x3 rotation matrix about the x-axis by angle alpha which is given it rad.
     *   - rotateAboutY(beta):  returns a 3x3 rotation matrix about the y-axis by angle beta which is given it rad.
     *   - rotateAboutZ(gamma): returns a 3x3 rotation matrix about the z-axis by angle gamma which is given it rad.
     *   - cardan(alpha, beta, gamma): returns a 3x3 rotation matrix of a cardan rotation about the angles alpha,
     *     beta and gamma which are given it rad.
     *   - euler(PHI, theta, phi): returns a 3x3 rotation matrix of a euler rotation about the angles PHI,
     *     theta and phi which are given it rad.
     *   - load(filename): returns a NxM matrix of the data stored in the file filename. filename may be a absolute
     *     relative path. A relative path is interprete relative to the location of the XML file with the load
     *     statement. (The abstract Eval class guarantees the the current path is at the XML file if load is called)
     */
    virtual boost::shared_ptr<void> callFunction(const std::string &name, const std::vector<boost::shared_ptr<void> >& args) const=0;

    static boost::shared_ptr<void> casadiValue;

    //! evaluate the string str using the current parameters and return the result.
    virtual boost::shared_ptr<void> fullStringToValue(const std::string &str, const xercesc::DOMElement *e=NULL) const=0;

    //! evaluate str partially and return result as an std::string
    std::string partialStringToString(const std::string &str, const xercesc::DOMElement *e) const;

    template<class E>
    static boost::shared_ptr<Eval> newEvaluator(std::vector<boost::filesystem::path>* dependencies_) {
      return boost::shared_ptr<E>(new E(dependencies_));
    }

    //! get the SWIG pointer of this value.
    virtual void* getSwigThis(const boost::shared_ptr<void> &value) const=0;
    //! get the SWIG type (class name) of this value.
    virtual std::string getSwigType(const boost::shared_ptr<void> &value) const=0;

  private:
    // virtual spezialization of cast(const boost::shared_ptr<void> &value)
    virtual double                            cast_double              (const boost::shared_ptr<void> &value) const=0;
    virtual std::vector<double>               cast_vector_double       (const boost::shared_ptr<void> &value) const=0;
    virtual std::vector<std::vector<double> > cast_vector_vector_double(const boost::shared_ptr<void> &value) const=0;
    virtual std::string                       cast_string              (const boost::shared_ptr<void> &value) const=0;
    // spezialization of cast(const boost::shared_ptr<void> &value)
    CodeString          cast_CodeString  (const boost::shared_ptr<void> &value) const;
    int                 cast_int         (const boost::shared_ptr<void> &value) const;

    // spezialization of cast(const boost::shared_ptr<void> &value, xercesc::DOMDocument *doc)
    xercesc::DOMElement* cast_DOMElement_p(const boost::shared_ptr<void> &value, xercesc::DOMDocument *doc) const;

    // virtual spezialization of create(...)
    virtual boost::shared_ptr<void> create_double              (const double& v) const=0;
    virtual boost::shared_ptr<void> create_vector_double       (const std::vector<double>& v) const=0;
    virtual boost::shared_ptr<void> create_vector_vector_double(const std::vector<std::vector<double> >& v) const=0;
    virtual boost::shared_ptr<void> create_string              (const std::string& v) const=0;

    boost::shared_ptr<void> handleUnit(const xercesc::DOMElement *e, const boost::shared_ptr<void> &ret);

    static std::map<std::string, std::string> units;

    static std::map<std::string, boost::function<boost::shared_ptr<Eval>(std::vector<boost::filesystem::path>*)> >& getEvaluators() {
      static std::map<std::string, boost::function<boost::shared_ptr<Eval>(std::vector<boost::filesystem::path>*)> > evaluators;
      return evaluators;
    };

};

template<typename T>
boost::shared_ptr<void> Eval::createSwig() const {
  return createSwigByTypeName(SwigType<T>::name);
}

template<typename T>
T Eval::cast(const boost::shared_ptr<void> &value) const {
  // treat all types T as a swig type
  if(getSwigType(value)!=SwigType<T>::name)
    throw DOMEvalException("This variable is not of SWIG type "+SwigType<T>::name+".");
  return static_cast<T>(getSwigThis(value));
}
// ... but prevere these specializations
template<> std::string Eval::cast<std::string>(const boost::shared_ptr<void> &value) const;
template<> CodeString Eval::cast<CodeString>(const boost::shared_ptr<void> &value) const;
template<> double Eval::cast<double>(const boost::shared_ptr<void> &value) const;
template<> int Eval::cast<int>(const boost::shared_ptr<void> &value) const;
template<> std::vector<double> Eval::cast<std::vector<double> >(const boost::shared_ptr<void> &value) const;
template<> std::vector<std::vector<double> > Eval::cast<std::vector<std::vector<double> > >(const boost::shared_ptr<void> &value) const;

// no template definition, only this spezialization
template<> xercesc::DOMElement* Eval::cast<xercesc::DOMElement*>(const boost::shared_ptr<void> &value, xercesc::DOMDocument *doc) const;

// spezializations for create
template<> boost::shared_ptr<void> Eval::create<double>                            (const double& v) const;
template<> boost::shared_ptr<void> Eval::create<std::vector<double> >              (const std::vector<double>& v) const;
template<> boost::shared_ptr<void> Eval::create<std::vector<std::vector<double> > >(const std::vector<std::vector<double> >& v) const;
template<> boost::shared_ptr<void> Eval::create<std::string>                       (const std::string& v) const;

} // end namespace MBXMLUtils

#endif
