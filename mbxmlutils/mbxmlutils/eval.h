#ifndef _MBXMLUTILS_EVAL_H_
#define _MBXMLUTILS_EVAL_H_

#include <fmatvec/atom.h>
#include <boost/filesystem.hpp>
#include <xercesc/util/XercesDefs.hpp>
#include <mbxmlutilshelper/dom.h>
#include <unordered_map>

#define MBXMLUTILS_EVAL_CONCAT1(X, Y) X##Y
#define MBXMLUTILS_EVAL_CONCAT(X, Y) MBXMLUTILS_EVAL_CONCAT1(X, Y)
#define MBXMLUTILS_EVAL_APPENDLINE(X) MBXMLUTILS_EVAL_CONCAT(X, __LINE__)

/** Use this macro to register a new evaluator */
#define MBXMLUTILS_EVAL_REGISTER(T) \
  namespace { \
    struct Reg { \
      Reg() { Eval::registerEvaluator<T>(); } \
    } MBXMLUTILS_EVAL_APPENDLINE(regDummy); \
  }

namespace XERCES_CPP_NAMESPACE {
  class DOMElement;
  class DOMAttr;
  class DOMDocument;
}

namespace MBXMLUtils {

bool tryDouble2Int(double d, int &i);

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
    NewParamLevel(std::shared_ptr<Eval> oe_, bool newLevel_=true);
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
    typedef std::shared_ptr<void> Value;

  protected:
    //! Constructor.
    Eval(std::vector<boost::filesystem::path> *dependencies_);
  public:
    //! Destructor.
    ~Eval() override;

    Eval(const Eval& other) = delete; // copy constructor
    Eval(Eval&& other) = delete; // move constructor
    Eval& operator=(const Eval& other) = delete; // copy assignment
    Eval& operator=(Eval&& other) = delete; // move assignment

    //! Create a evaluator.
    static std::shared_ptr<Eval> createEvaluator(const std::string &evalName, std::vector<boost::filesystem::path> *dependencies_=nullptr);

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
    Value eval(const xercesc::DOMAttr *a);

    /*! Cast the value value to type <tt>T</tt>.
     * Possible combinations of allowed value types and template types <tt>T</tt> are listed in the
     * following table. If a combination is not allowed a exception is thrown.
     * <table>
     *   <tr>
     *     <th></th>
     *     <th colspan="5"><tt>value</tt> is of Type ...</th>
     *   </tr>
     *   <tr>
     *     <th>Template Type <tt>T</tt> equals ...</th>
     *     <th>real scalar</th>
     *     <th>real vector</th>
     *     <th>real matrix</th>
     *     <th>string</th>
     *     <th><tt>Function</tt></th>
     *   </tr>
     *
     *   <tr>
     *     <!--CAST TO-->    <th><tt>int</tt></th>
     *     <!--scalar-->     <td>only if its integral number</td>
     *     <!--vector-->     <td></td>
     *     <!--matrix-->     <td></td>
     *     <!--string-->     <td></td>
     *     <!--Function-->   <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>double</tt></th>
     *     <!--scalar-->     <td>X</td>
     *     <!--vector-->     <td></td>
     *     <!--matrix-->     <td></td>
     *     <!--string-->     <td></td>
     *     <!--Function-->   <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>vector&lt;double&gt;</tt></th>
     *     <!--scalar-->     <td>X</td>
     *     <!--vector-->     <td>X</td>
     *     <!--matrix-->     <td></td>
     *     <!--string-->     <td></td>
     *     <!--Function-->   <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>vector&lt;vector&lt;double&gt;&nbsp;&gt;</tt></th>
     *     <!--scalar-->     <td>X</td>
     *     <!--vector-->     <td>X</td>
     *     <!--matrix-->     <td>X</td>
     *     <!--string-->     <td></td>
     *     <!--Function-->   <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>string</tt></th>
     *     <!--scalar-->     <td></td>
     *     <!--vector-->     <td></td>
     *     <!--matrix-->     <td></td>
     *     <!--string-->     <td>X</td>
     *     <!--Function-->   <td></td>
     *   </tr>
     *   <tr>
     *     <!--CAST TO-->    <th><tt>CodeString</tt></th>
     *     <!--scalar-->     <td>returns e.g. <code>5</code></td>
     *     <!--vector-->     <td>returns e.g. <code>[3;7]</code></td>
     *     <!--matrix-->     <td>returns e.g. <code>[1,3;5,4]</code></td>
     *     <!--string-->     <td>returns e.g. <code>'foo'</code></td>
     *     <!--Function-->   <td>returns e.g. <code>( 2 [...] {...} [...])</code></td>
     *   </tr>
     * </table>
     */
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
    Value stringToValue(const std::string &str, const xercesc::DOMElement *e=nullptr, bool fullEval=true) const;

    //! create a value of the given type. T can be one of:
    //! double: create a floating point value
    //! vector<double>: create a vector of floating point values
    //! vector<vector<double>>: create a matrix of floating point values
    //! string: create a string value
    //! vector<Value>: create a vector of dependent functions
    //! vector<vector<Value>: create a matrix of dependent functions
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
    //! Add a independent variable named paramName and return Value.
    virtual Value addFunctionIndepParam(const std::string &paramName, int dim) = 0;

    //! create a Function with n independents and a dependent function (scalar, vector or matrix)
    virtual Value createFunction(const std::vector<Value> &indeps, const Value &dep) const = 0;

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
    Value currentImport;
    // stack of imports
    std::stack<Value> importStack;

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

    //! evaluate the string str using the current parameters and return the result.
    virtual Value fullStringToValue(const std::string &str, const xercesc::DOMElement *e) const=0;

    //! evaluate str partially and return result as an std::string
    std::string partialStringToString(const std::string &str, const xercesc::DOMElement *e) const;

    template<class E>
    static std::shared_ptr<Eval> newEvaluator(std::vector<boost::filesystem::path>* dependencies_) {
      return std::shared_ptr<E>(new E(dependencies_));
    }

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

    // virtual spezialization of create(...)
    virtual Value create_double                   (const double& v) const=0;
    virtual Value create_vector_double            (const std::vector<double>& v) const=0;
    virtual Value create_vector_vector_double     (const std::vector<std::vector<double> >& v) const=0;
    virtual Value create_string                   (const std::string& v) const=0;
    virtual Value create_vector_FunctionDep       (const std::vector<Value>& v) const=0;
    virtual Value create_vector_vector_FunctionDep(const std::vector<std::vector<Value> >& v) const=0;

    virtual std::string serializeFunction(const Value &x) const = 0;

    Value handleUnit(const xercesc::DOMElement *e, const Value &ret);

    static std::map<std::string, std::string> units;

    static std::map<std::string, std::function<std::shared_ptr<Eval>(std::vector<boost::filesystem::path>*)> >& getEvaluators();

};

// specializations
template<> std::string Eval::cast<std::string>(const Value &value) const;
template<> CodeString Eval::cast<CodeString>(const Value &value) const;
template<> double Eval::cast<double>(const Value &value) const;
template<> int Eval::cast<int>(const Value &value) const;
template<> std::vector<double> Eval::cast<std::vector<double> >(const Value &value) const;
template<> std::vector<std::vector<double> > Eval::cast<std::vector<std::vector<double> > >(const Value &value) const;

// spezializations for create
template<> Eval::Value Eval::create<double>                               (const double& v) const;
template<> Eval::Value Eval::create<std::vector<double> >                 (const std::vector<double>& v) const;
template<> Eval::Value Eval::create<std::vector<std::vector<double> > >   (const std::vector<std::vector<double> >& v) const;
template<> Eval::Value Eval::create<std::string>                          (const std::string& v) const;
template<> Eval::Value Eval::create<std::vector<Eval::Value>>             (const std::vector<Value>& v) const;
template<> Eval::Value Eval::create<std::vector<std::vector<Eval::Value>>>(const std::vector<std::vector<Value> >& v) const;

} // end namespace MBXMLUtils

#endif
