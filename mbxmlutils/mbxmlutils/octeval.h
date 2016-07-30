#ifndef _MBXMLUTILS_OCTEVAL_H_
#define _MBXMLUTILS_OCTEVAL_H_

#include "eval.h"
#include <boost/scoped_ptr.hpp>
#include <memory>
#include <boost/make_shared.hpp>

class octave_value;
class octave_value_list;
class octave_function;

namespace XERCES_CPP_NAMESPACE { class DOMElement; }

namespace MBXMLUtils {

std::shared_ptr<octave_value> C(const std::shared_ptr<void> &value);

std::shared_ptr<void> C(const octave_value &value);

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
    static std::string getNameStatic() { return "octave"; }
    std::string getName() const { return getNameStatic(); }

    //! Add octave search path to the current evaluator context.
    //! \p code must evaluate to a string representing a directory/path.
    //! A relative path is expanded to an absolute path using the path of e as current directory.
    //! The absolute path is then added using "addpath" to the octave search path.
    void addImport(const std::string &code, const xercesc::DOMElement *e, bool deprecated=false);

    //! get the type of value
    bool valueIsOfType(const std::shared_ptr<void> &value, ValueType type) const;

    //! return a list of all required files of octave (excluding dependent files of libraries)
    std::map<boost::filesystem::path, std::pair<boost::filesystem::path, bool> >& requiredFiles() const;

    virtual bool useOneBasedIndexes() { return true; }

  protected:

    //! evaluate str fully and return result as an octave variable
    virtual std::shared_ptr<void> fullStringToValue(const std::string &str, const xercesc::DOMElement *e=NULL) const;

    //! get the SWIG pointer of this value.
    void* getSwigThis(const std::shared_ptr<void> &value) const;

    //! get the SWIG class name of this value.
    std::string getSwigType(const std::shared_ptr<void> &value) const;

    std::shared_ptr<void> createSwigByTypeName(const std::string &typeName) const;

    static octave_value_list fevalThrow(octave_function *func, const octave_value_list &arg, int n=0,
                                        const std::string &msg=std::string());

    virtual std::shared_ptr<void> callFunction(const std::string &name, const std::vector<std::shared_ptr<void> >& args) const;

    virtual double                            cast_double              (const std::shared_ptr<void> &value) const;
    virtual std::vector<double>               cast_vector_double       (const std::shared_ptr<void> &value) const;
    virtual std::vector<std::vector<double> > cast_vector_vector_double(const std::shared_ptr<void> &value) const;
    virtual std::string                       cast_string              (const std::shared_ptr<void> &value) const;

    virtual std::shared_ptr<void> create_double              (const double& v) const;
    virtual std::shared_ptr<void> create_vector_double       (const std::vector<double>& v) const;
    virtual std::shared_ptr<void> create_vector_vector_double(const std::vector<std::vector<double> >& v) const;
    virtual std::shared_ptr<void> create_string              (const std::string& v) const;
};

} // end namespace MBXMLUtils

#endif
