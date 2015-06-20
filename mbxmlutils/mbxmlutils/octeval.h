#ifndef _MBXMLUTILS_OCTEVAL_H_
#define _MBXMLUTILS_OCTEVAL_H_

#include "eval.h"
#include <boost/scoped_ptr.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

class octave_value;
class octave_value_list;
class octave_function;

namespace XERCES_CPP_NAMESPACE { class DOMElement; }

namespace MBXMLUtils {

boost::shared_ptr<octave_value> C(const boost::shared_ptr<void> &value);

boost::shared_ptr<void> C(const octave_value &value);

class OctEval;

/*! Octave expression evaluator and converter. */
class OctEval : public Eval {
  friend class Eval;

  protected:
    //! Constructor.
    OctEval(std::vector<boost::filesystem::path> *dependencies_=NULL);
  public:
    //! Destructor.
    ~OctEval();

    //! Get the name of this evaluator.
    std::string getEvaluatorName() {
      return "octave";
    }

    //! Add dir to octave search path
    //! A relative path in dir is expanded to an absolute path using the current directory.
    void addPath(const boost::filesystem::path &dir, const xercesc::DOMElement *e);
    
    //! Evaluate the XML element e using the current parameters returning the resulting octave value.
    //! The type of evaluation depends on the type of e.
    boost::shared_ptr<void> eval(const xercesc::DOMElement *e);

    //! Evaluate the XML attribute a using the current parameters returning the resulting octave value.
    //! The type of evaluation depends on the type of a.
    //! The result of a "partially" evaluation is returned as a octave string even so it is not really a string.
    boost::shared_ptr<void> eval(const xercesc::DOMAttr *a, const xercesc::DOMElement *pe=NULL);

    //! get the type of value
    ValueType getType(const boost::shared_ptr<void> &value);

    //! evaluate str and return result as an octave variable, this can be used to evaluate outside of XML.
    //! If e is given it is used as location information in case of errors.
    //! If fullEval is false the "partially" evaluation is returned as a octave string even so it is not really a string.
    boost::shared_ptr<void> stringToValue(const std::string &str, const xercesc::DOMElement *e=NULL, bool fullEval=true);

    //! return a list of all required files of octave (excluding dependent files of libraries)
    std::map<boost::filesystem::path, std::pair<boost::filesystem::path, bool> >& requiredFiles();

  protected:

    //! This function deinitialized octave. It is used in the dtor and before exceptions in the ctor are thrown
    // (in the later case the dtor is not called but octave must be uninitialized before exit)
    void deinitOctave();

    //! evaluate str fully and return result as an octave variable
    octave_value fullStringToOctValue(const std::string &str, const xercesc::DOMElement *e=NULL);

    //! evaluate str partially and return result as an std::string
    std::string partialStringToOctValue(const std::string &str, const xercesc::DOMElement *e);

    //! cast value to the corresponding swig object of type T, without ANY type check.
    void* castToSwig(const boost::shared_ptr<void> &value);

    //! create octave value of CasADi type name. Created using the default ctor.
    static octave_value createCasADi(const std::string &name);

    // initial path
    static std::string initialPath;
    static std::string pathSep;

    static int initCount;

    static std::map<std::string, std::string> units;

    static boost::scoped_ptr<octave_value> casadiOctValue;

    static octave_value_list fevalThrow(octave_function *func, const octave_value_list &arg, int n=0,
                                        const std::string &msg=std::string());

    boost::shared_ptr<octave_value> handleUnit(const xercesc::DOMElement *e, const boost::shared_ptr<octave_value> &ret);

    virtual int                               cast_int                 (const boost::shared_ptr<void> &value);
    virtual double                            cast_double              (const boost::shared_ptr<void> &value);
    virtual std::vector<double>               cast_vector_double       (const boost::shared_ptr<void> &value);
    virtual std::vector<std::vector<double> > cast_vector_vector_double(const boost::shared_ptr<void> &value);
    virtual std::string                       cast_string              (const boost::shared_ptr<void> &value);
    virtual casadi::SXFunction                cast_SXFunction          (const boost::shared_ptr<void> &value);
    virtual casadi::SXFunction*               cast_SXFunction_p        (const boost::shared_ptr<void> &value);
    virtual casadi::SX                        cast_SX                  (const boost::shared_ptr<void> &value);
    virtual casadi::SX*                       cast_SX_p                (const boost::shared_ptr<void> &value);
    virtual casadi::DMatrix                   cast_DMatrix             (const boost::shared_ptr<void> &value);
    virtual casadi::DMatrix*                  cast_DMatrix_p           (const boost::shared_ptr<void> &value);
    virtual xercesc::DOMElement*              cast_DOMElement_p        (const boost::shared_ptr<void> &value, xercesc::DOMDocument *doc);

    virtual boost::shared_ptr<void> create_double              (const double& v);
    virtual boost::shared_ptr<void> create_vector_double       (const std::vector<double>& v);
    virtual boost::shared_ptr<void> create_vector_vector_double(const std::vector<std::vector<double> >& v);
    virtual boost::shared_ptr<void> create_string              (const std::string& v);
};

} // end namespace MBXMLUtils

#endif
