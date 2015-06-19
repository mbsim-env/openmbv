#ifndef _MBXMLUTILS_OCTEVAL_H_
#define _MBXMLUTILS_OCTEVAL_H_

#include "eval.h"
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

inline boost::shared_ptr<octave_value> C(const boost::shared_ptr<void> &value) {
  return boost::static_pointer_cast<octave_value>(value);
}

inline boost::shared_ptr<void> C(const octave_value &value) {
  return boost::make_shared<octave_value>(value);
}

class OctEval;

/*! Octave expression evaluator and converter. */
class OctEval : public Eval {
  public:
    //! Constructor.
    OctEval(std::vector<boost::filesystem::path> *dependencies_=NULL);
    //! Destructor.
    ~OctEval();

    //! Add a octave value to the current parameters.
    void addParam(const std::string &paramName, const boost::shared_ptr<void>& value);
    //! Add all parameters from XML element e.
    //! The parameters are added from top to bottom as they appear in the XML element e.
    //! Parameters may depend on parameters already added.
    void addParamSet(const xercesc::DOMElement *e);

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
};

} // end namespace MBXMLUtils

#endif
