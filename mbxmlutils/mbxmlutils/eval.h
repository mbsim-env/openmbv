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

extern bool deactivateBlock;

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
    Eval(std::vector<boost::filesystem::path> *dependencies_=NULL);
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
    virtual std::string getName()=0;

    //! Add a value to the current parameters.
    void addParam(const std::string &paramName, const boost::shared_ptr<void>& value);
    //! Add all parameters from XML element e.
    //! The parameters are added from top to bottom as they appear in the XML element e.
    //! Parameters may depend on parameters already added.
    void addParamSet(const xercesc::DOMElement *e);

    //! Add dir to the evauator search path.
    //! A relative path in dir is expanded to an absolute path using the current directory.
    virtual void addPath(const boost::filesystem::path &dir, const xercesc::DOMElement *e)=0;//MFMF this is not portable for different evaluators
    
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
    T cast(const boost::shared_ptr<void> &value, xercesc::DOMDocument *doc);

    //! see cast(const boost::shared_ptr<void> &value, shared_ptr<DOMDocument> &doc)
    template<typename T>
    T cast(const boost::shared_ptr<void> &value);

    //! check whether value is of type type.
    //! Note that true is only returned if the value is really of type type. If value can be casted
    //! to type type but is not of this type then false is returned.
    //! Note that their are evaluators (e.g. octave) which cannot distinguish between e.g. a scalar,
    //! a vector of size 1 or a matrix of size 1x1. Hence, these evaluators will return true for ScalarType
    //! in all these three cases and analog for VectorType.
    virtual bool valueIsOfType(const boost::shared_ptr<void> &value, ValueType type)=0;

    //! evaluate str and return result as an variable, this can be used to evaluate outside of XML.
    //! If e is given it is used as location information in case of errors.
    //! If fullEval is false the "partially" evaluation is returned as a string even so it is not really a string.
    boost::shared_ptr<void> stringToValue(const std::string &str, const xercesc::DOMElement *e=NULL, bool fullEval=true);

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

    template<typename T>
    boost::shared_ptr<void> createSwig();

    virtual boost::shared_ptr<void> createSwigByTypeName(const std::string &typeName)=0;

    virtual boost::shared_ptr<void> callFunction(const std::string &name, const std::vector<boost::shared_ptr<void> >& args)=0;

    static boost::shared_ptr<void> casadiValue;

    virtual boost::shared_ptr<void> fullStringToValue(const std::string &str, const xercesc::DOMElement *e=NULL)=0;

    //! evaluate str partially and return result as an std::string
    std::string partialStringToString(const std::string &str, const xercesc::DOMElement *e);

    template<class E>
    static boost::shared_ptr<Eval> newEvaluator(std::vector<boost::filesystem::path>* dependencies_) {
      return boost::shared_ptr<E>(new E(dependencies_));
    }

    //! get the SWIG pointer of this value.
    virtual void* getSwigThis(const boost::shared_ptr<void> &value)=0;
    //! get the SWIG type (class name) of this value.
    virtual std::string getSwigType(const boost::shared_ptr<void> &value)=0;

  private:
    // virtual spezialization of cast(const boost::shared_ptr<void> &value)
    virtual double                            cast_double              (const boost::shared_ptr<void> &value)=0;
    virtual std::vector<double>               cast_vector_double       (const boost::shared_ptr<void> &value)=0;
    virtual std::vector<std::vector<double> > cast_vector_vector_double(const boost::shared_ptr<void> &value)=0;
    virtual std::string                       cast_string              (const boost::shared_ptr<void> &value)=0;
    // spezialization of cast(const boost::shared_ptr<void> &value)
    CodeString          cast_CodeString  (const boost::shared_ptr<void> &value);
    int                 cast_int         (const boost::shared_ptr<void> &value);

    // spezialization of cast(const boost::shared_ptr<void> &value, xercesc::DOMDocument *doc)
    xercesc::DOMElement* cast_DOMElement_p(const boost::shared_ptr<void> &value, xercesc::DOMDocument *doc);

    // virtual spezialization of create(...)
    virtual boost::shared_ptr<void> create_double              (const double& v)=0;
    virtual boost::shared_ptr<void> create_vector_double       (const std::vector<double>& v)=0;
    virtual boost::shared_ptr<void> create_vector_vector_double(const std::vector<std::vector<double> >& v)=0;
    virtual boost::shared_ptr<void> create_string              (const std::string& v)=0;

    boost::shared_ptr<void> handleUnit(const xercesc::DOMElement *e, const boost::shared_ptr<void> &ret);

    static std::map<std::string, std::string> units;

    static std::map<std::string, boost::function<boost::shared_ptr<Eval>(std::vector<boost::filesystem::path>*)> >& getEvaluators() {
      static std::map<std::string, boost::function<boost::shared_ptr<Eval>(std::vector<boost::filesystem::path>*)> > evaluators;
      return evaluators;
    };

};

template<typename T>
boost::shared_ptr<void> Eval::createSwig() {
  return createSwigByTypeName(SwigType<T>::name);
}

template<typename T>
T Eval::cast(const boost::shared_ptr<void> &value) {
  // treat all types T as a swig type
  if(getSwigType(value)!=SwigType<T>::name)
    throw DOMEvalException("This variable is not of SWIG type "+SwigType<T>::name+".");
  return static_cast<T>(getSwigThis(value));
}
// ... but prevere these specializations
template<> std::string Eval::cast<std::string>(const boost::shared_ptr<void> &value);
template<> CodeString Eval::cast<CodeString>(const boost::shared_ptr<void> &value);
template<> double Eval::cast<double>(const boost::shared_ptr<void> &value);
template<> int Eval::cast<int>(const boost::shared_ptr<void> &value);
template<> std::vector<double> Eval::cast<std::vector<double> >(const boost::shared_ptr<void> &value);
template<> std::vector<std::vector<double> > Eval::cast<std::vector<std::vector<double> > >(const boost::shared_ptr<void> &value);

// no template definition, only this spezialization
template<> xercesc::DOMElement* Eval::cast<xercesc::DOMElement*>(const boost::shared_ptr<void> &value, xercesc::DOMDocument *doc);

// spezializations for create
template<> boost::shared_ptr<void> Eval::create<double>                            (const double& v);
template<> boost::shared_ptr<void> Eval::create<std::vector<double> >              (const std::vector<double>& v);
template<> boost::shared_ptr<void> Eval::create<std::vector<std::vector<double> > >(const std::vector<std::vector<double> >& v);
template<> boost::shared_ptr<void> Eval::create<std::string>                       (const std::string& v);

} // end namespace MBXMLUtils

#endif
