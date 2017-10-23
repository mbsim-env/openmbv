#ifndef _MBXMLUTILS_EVAL_H_
#define _MBXMLUTILS_EVAL_H_

#include <fmatvec/atom.h>
#include <boost/filesystem.hpp>
#include <boost/variant.hpp>
#include <xercesc/util/XercesDefs.hpp>
#include <mbxmlutilshelper/dom.h>
#include <casadi/casadi.hpp>
#include <unordered_map>

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
    NewParamLevel(const std::shared_ptr<Eval> &oe_, bool newLevel_=true);
    //! Reset to the previous parameter level
    ~NewParamLevel();
    
    NewParamLevel(const NewParamLevel& other) = delete; // copy constructor
    NewParamLevel(NewParamLevel&& other) = delete; // move constructor
    NewParamLevel& operator=(const NewParamLevel& other) = delete; // copy assignment
    NewParamLevel& operator=(NewParamLevel&& other) = delete; // move assignment
  protected:
    static void* operator new(std::size_t); // no heap allocation allowed
    static void* operator new[](std::size_t); // no heap allocation allowed

    std::shared_ptr<Eval> oe;
    bool newLevel;
};

template<class T> struct SwigType { static std::string name; };

/*! Expression evaluator and converter. */
class Eval : public std::enable_shared_from_this<Eval>, virtual public fmatvec::Atom {
  public:
    friend class NewParamLevel;

    //! Known types for a variable
    //! See also valueIsOfType.
    enum ValueType {
      ScalarType,
      VectorType,
      MatrixType,
      StringType,
      FunctionType
    };

    //! Typedef for a shared value
    typedef std::pair<std::vector<casadi::SX>, std::vector<casadi::SX>> Function;
    typedef boost::variant<std::shared_ptr<void>, Function> Value;

  protected:
    //! Constructor.
    Eval(std::vector<boost::filesystem::path> *dependencies_);
  public:
    //! Destructor.
    ~Eval();

    Eval(const Eval& other) = delete; // copy constructor
    Eval(Eval&& other) = delete; // move constructor
    Eval& operator=(const Eval& other) = delete; // copy assignment
    Eval& operator=(Eval&& other) = delete; // move assignment

    //! Create a evaluator.
    static std::shared_ptr<Eval> createEvaluator(const std::string &evalName, std::vector<boost::filesystem::path> *dependencies_=NULL);

    // Register a new evaluator.
    template<class E>
    static void registerEvaluator() {
      getEvaluators()[E::getNameStatic()]=&newEvaluator<E>;
    }

    //! Get the type of this evaluator
    virtual std::string getName() const=0;

    //! Add a value to the current parameters.
    void addParam(const std::string &paramName, const Value& value);
    //! Add all parameters from XML element e.
    //! The parameters are added from top to bottom as they appear in the XML element e.
    //! Parameters may depend on parameters already added.
    void addParamSet(const xercesc::DOMElement *e);

    //! Import evaluator statements. This routine highly depends on the evaluator.
    //! See the spezialized evaluators documentation for details.
    virtual void addImport(const std::string &code, const xercesc::DOMElement *e)=0;
    
    //! Evaluate the XML element e using the current parameters returning the resulting value.
    //! The type of evaluation depends on the type of e.
    Value eval(const xercesc::DOMElement *e);

    //! Evaluate the XML attribute a using the current parameters returning the resulting value.
    //! The type of evaluation depends on the type of a.
    //! The result of a "partially" evaluation is returned as a string even so it is not really a string.
    Value eval(const xercesc::DOMAttr *a, const xercesc::DOMElement *pe=NULL);

    /*! Cast the value value to type <tt>T</tt>.
     * Possible combinations of allowed value types and template types <tt>T</tt> are listed in the
     * following table. If a combination is not allowed a exception is thrown.
     * If a c-pointer is returned this c-pointer is only guaranteed to be valid for the lifetime of the object
     * \p value being passed as argument. This pointer is a reference to the value \p value in the interpreter.
     * If a DOMElement* is returned \p doc must be given and ownes the memory of the returned DOM tree. For other
     * return types this function must be called with only one argument cast(const Value &value);
     * <table>
     *   <tr>
     *     <th></th>
     *     <th colspan="7"><tt>value</tt> is of Type ...</th>
     *   </tr>
     *   <tr>
     *     <th>Template Type <tt>T</tt> equals ...</th>
     *     <th>real scalar</th>
     *     <th>real vector</th>
     *     <th>real matrix</th>
     *     <th>string</th>
     *     <th><i>SWIG</i> <tt>casadi::SX</tt></th>
     *     <th><tt>Function</tt></th>
     *     <th><i>SWIG</i> <tt>XYZ</tt></th>
     *   </tr>
     *
     *   <tr>
     *     <!--CAST TO-->    <th><tt>int</tt></th>
     *     <!--scalar-->     <td>only if its integral number</td>
     *     <!--vector-->     <td></td>
     *     <!--matrix-->     <td></td>
     *     <!--string-->     <td></td>
     *     <!--SX-->         <td></td>
     *     <!--Function-->   <td></td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>double</tt></th>
     *     <!--scalar-->     <td>X</td>
     *     <!--vector-->     <td></td>
     *     <!--matrix-->     <td></td>
     *     <!--string-->     <td></td>
     *     <!--SX-->         <td></td>
     *     <!--Function-->   <td></td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>vector&lt;double&gt;</tt></th>
     *     <!--scalar-->     <td>X</td>
     *     <!--vector-->     <td>X</td>
     *     <!--matrix-->     <td></td>
     *     <!--string-->     <td></td>
     *     <!--SX-->         <td></td>
     *     <!--Function-->   <td></td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>vector&lt;vector&lt;double&gt;&nbsp;&gt;</tt></th>
     *     <!--scalar-->     <td>X</td>
     *     <!--vector-->     <td>X</td>
     *     <!--matrix-->     <td>X</td>
     *     <!--string-->     <td></td>
     *     <!--SX-->         <td></td>
     *     <!--Function-->   <td></td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>string</tt></th>
     *     <!--scalar-->     <td></td>
     *     <!--vector-->     <td></td>
     *     <!--matrix-->     <td></td>
     *     <!--string-->     <td>X</td>
     *     <!--SX-->         <td></td>
     *     <!--Function-->   <td></td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>CodeString</tt></th>
     *     <!--scalar-->     <td>returns e.g. <code>5</code></td>
     *     <!--vector-->     <td>returns e.g. <code>[3;7]</code></td>
     *     <!--matrix-->     <td>returns e.g. <code>[1,3;5,4]</code></td>
     *     <!--string-->     <td>returns e.g. <code>'foo'</code></td>
     *     <!--SX-->         <td></td>
     *     <!--Function-->   <td></td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>DOMElement*</tt></th>
     *     <!--scalar-->     <td></td>
     *     <!--vector-->     <td></td>
     *     <!--matrix-->     <td></td>
     *     <!--string-->     <td></td>
     *     <!--SX-->         <td></td>
     *     <!--Function-->   <td>X</td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>casadi::SX</tt></th>
     *     <!--scalar-->     <td>X</td>
     *     <!--vector-->     <td>X</td>
     *     <!--matrix-->     <td>X</td>
     *     <!--string-->     <td></td>
     *     <!--SX-->         <td>X</td>
     *     <!--Function-->   <td></td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>casadi::SX*</tt></th>
     *     <!--scalar-->     <td></td>
     *     <!--vector-->     <td></td>
     *     <!--matrix-->     <td></td>
     *     <!--string-->     <td></td>
     *     <!--SX-->         <td>X</td>
     *     <!--Function-->   <td></td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>Function</tt></th>
     *     <!--scalar-->     <td></td>
     *     <!--vector-->     <td></td>
     *     <!--matrix-->     <td></td>
     *     <!--string-->     <td></td>
     *     <!--SX-->         <td></td>
     *     <!--Function-->   <td>X</td>
     *     <!--XYZ-->        <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>XYZ*</tt></th>
     *     <!--scalar-->     <td></td>
     *     <!--vector-->     <td></td>
     *     <!--matrix-->     <td></td>
     *     <!--string-->     <td></td>
     *     <!--SX-->         <td></td>
     *     <!--Function-->   <td></td>
     *     <!--XYZ-->        <td>*</td>
     *   </tr>
     * </table>
     * For arbitary evaluator values of type SWIG wrapper object (see * in table) a instantiation of
     * <code>template<> string SwigType<XYZ*>::name("<swig_type_name_of_XYZ>");</code>
     * is required.
     */
    template<typename T>
    T cast(const Value &value, xercesc::DOMDocument *doc) const;

    //! see cast(const Value &value, shared_ptr<DOMDocument> &doc)
    template<typename T>
    T cast(const Value &value) const;

    //! check whether value is of type type.
    //! Note that true is only returned if the value is really of type type. If value can be casted
    //! to type type but is not of this type then false is returned.
    //! Note that their are evaluators (e.g. octave) which cannot distinguish between e.g. a scalar,
    //! a vector of size 1 or a matrix of size 1x1. Hence, these evaluators will return true for ScalarType
    //! in all these three cases and analog for VectorType.
    virtual bool valueIsOfType(const Value &value, ValueType type) const=0;

    //! evaluate str and return result as an variable, this can be used to evaluate outside of XML.
    //! If e is given it is used as location information in case of errors.
    //! If fullEval is false the "partially" evaluation is returned as a string even so it is not really a string.
    Value stringToValue(const std::string &str, const xercesc::DOMElement *e=NULL, bool fullEval=true) const;

    //! create a value of the given type
    template<class T>
    Value create(const T& v) const;

    //! return a list of all required files of the evaluator (excluding dependent files of libraries)
    virtual std::map<boost::filesystem::path, std::pair<boost::filesystem::path, bool> >& requiredFiles() const=0;

    //! convert a index (scalar or vector).
    //! if evalTo1Based == true: convert from the script language 0/1 base to 1 base.
    //! if evalTo1Based == false: convert from 1 base to the script language 0/1 base.
    virtual void convertIndex(Value &v, bool evalTo1Based)=0;

    //! Set value on DOMElement (is used by Eval::cast)
    static void setValue(xercesc::DOMElement *e, const Value &v);

  protected:
    //! Push the current context to a internal stack.
    void pushContext();

    //! Overwrite the current context with the top level context from the internal stack.
    void popContext();

    std::vector<boost::filesystem::path> *dependencies;

    // map of the current parameters
    std::unordered_map<std::string, Value> currentParam;
    // stack of parameters
    std::stack<std::unordered_map<std::string, Value> > paramStack;

    // current imports
    std::shared_ptr<void> currentImport;
    // stack of imports
    std::stack<std::shared_ptr<void> > importStack;

    template<typename T>
    Value createSwig() const;

    //! create a SWIG object of name typeName.
    virtual Value createSwigByTypeName(const std::string &typeName) const=0;

    /*! Return the value of a call to name using the arguments args.
     * The following functions must be implemented by the evaluator:
     *   - rotateAboutX(alpha): returns a 3x3 rotation matrix about the x-axis by angle alpha which is given in rad.
     *   - rotateAboutY(beta):  returns a 3x3 rotation matrix about the y-axis by angle beta which is given in rad.
     *   - rotateAboutZ(gamma): returns a 3x3 rotation matrix about the z-axis by angle gamma which is given in rad.
     *   - cardan(alpha, beta, gamma): returns a 3x3 rotation matrix of a cardan rotation about the angles alpha,
     *     beta and gamma which are given in rad.
     *   - euler(PHI, theta, phi): returns a 3x3 rotation matrix of a euler rotation about the angles PHI,
     *     theta and phi which are given in rad.
     *   - load(filename): returns a NxM matrix of the data stored in the file filename. filename may be a absolute
     *     or relative path. A relative path is interprete relative to the location of the XML file with the load
     *     statement. (The abstract Eval class guarantees that the current path is at the XML file if load is called)
     */
    virtual Value callFunction(const std::string &name, const std::vector<Value>& args) const=0;

    Value casadiValue;

    //! evaluate the string str using the current parameters and return the result.
    virtual Value fullStringToValue(const std::string &str, const xercesc::DOMElement *e=NULL) const=0;

    //! evaluate str partially and return result as an std::string
    std::string partialStringToString(const std::string &str, const xercesc::DOMElement *e) const;

    template<class E>
    static std::shared_ptr<Eval> newEvaluator(std::vector<boost::filesystem::path>* dependencies_) {
      return std::shared_ptr<E>(new E(dependencies_));
    }

    //! get the SWIG pointer of this value.
    virtual void* getSwigThis(const Value &value) const=0;
    //! get the SWIG type (class name) of this value.
    virtual std::string getSwigType(const Value &value) const=0;

    void addStaticDependencies(const xercesc::DOMElement *e) const;

  private:
    // virtual spezialization of cast(const Value &value)
    virtual double                            cast_double              (const Value &value) const=0;
    virtual std::vector<double>               cast_vector_double       (const Value &value) const=0;
    virtual std::vector<std::vector<double> > cast_vector_vector_double(const Value &value) const=0;
    virtual std::string                       cast_string              (const Value &value) const=0;
    // spezialization of cast(const Value &value)
    CodeString          cast_CodeString  (const Value &value) const;
    int                 cast_int         (const Value &value) const;
    casadi::SX          cast_SX          (const Value &value) const;

    // spezialization of cast(const Value &value, xercesc::DOMDocument *doc)
    xercesc::DOMElement* cast_DOMElement_p(const Value &value, xercesc::DOMDocument *doc) const;

    // virtual spezialization of create(...)
    virtual Value create_double              (const double& v) const=0;
    virtual Value create_vector_double       (const std::vector<double>& v) const=0;
    virtual Value create_vector_vector_double(const std::vector<std::vector<double> >& v) const=0;
    virtual Value create_string              (const std::string& v) const=0;

    Value handleUnit(const xercesc::DOMElement *e, const Value &ret);

    static std::map<std::string, std::string> units;

    static std::map<std::string, std::function<std::shared_ptr<Eval>(std::vector<boost::filesystem::path>*)> >& getEvaluators();

    void convertQName(Value &ret, const xercesc::DOMElement *e, const xercesc::DOMAttr *a);

};

template<typename T>
Eval::Value Eval::createSwig() const {
  return createSwigByTypeName(SwigType<T>::name);
}

template<typename T>
T Eval::cast(const Value &value) const {
  // treat all types T as a swig type
  if(getSwigType(value)!=SwigType<T>::name)
    throw DOMEvalException("This variable is not of SWIG type "+SwigType<T>::name+".");
  return static_cast<T>(getSwigThis(value));
}
// ... but prevere these specializations
template<> std::string Eval::cast<std::string>(const Value &value) const;
template<> CodeString Eval::cast<CodeString>(const Value &value) const;
template<> double Eval::cast<double>(const Value &value) const;
template<> int Eval::cast<int>(const Value &value) const;
template<> std::vector<double> Eval::cast<std::vector<double> >(const Value &value) const;
template<> std::vector<std::vector<double> > Eval::cast<std::vector<std::vector<double> > >(const Value &value) const;
template<> casadi::SX Eval::cast<casadi::SX>(const Value &value) const;
template<> Eval::Function Eval::cast<Eval::Function>(const Value &value) const;

// no template definition, only this spezialization
template<> xercesc::DOMElement* Eval::cast<xercesc::DOMElement*>(const Value &value, xercesc::DOMDocument *doc) const;

// spezializations for create
template<> Eval::Value Eval::create<double>                            (const double& v) const;
template<> Eval::Value Eval::create<std::vector<double> >              (const std::vector<double>& v) const;
template<> Eval::Value Eval::create<std::vector<std::vector<double> > >(const std::vector<std::vector<double> >& v) const;
template<> Eval::Value Eval::create<std::string>                       (const std::string& v) const;

} // end namespace MBXMLUtils

#endif
