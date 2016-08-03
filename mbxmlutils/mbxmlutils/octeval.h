#ifndef _MBXMLUTILS_OCTEVAL_H_
#define _MBXMLUTILS_OCTEVAL_H_

#include "eval.h"
#include <memory>

class octave_value;
class octave_value_list;
class octave_function;

namespace XERCES_CPP_NAMESPACE { class DOMElement; }

namespace MBXMLUtils {

std::shared_ptr<octave_value> C(const Eval::Value &value);

Eval::Value C(const octave_value &value);

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
    bool valueIsOfType(const Value &value, ValueType type) const;

    //! return a list of all required files of octave (excluding dependent files of libraries)
    std::map<boost::filesystem::path, std::pair<boost::filesystem::path, bool> >& requiredFiles() const;

    virtual bool useOneBasedIndexes() { return true; }

  protected:

    //! evaluate str fully and return result as an octave variable
    virtual Value fullStringToValue(const std::string &str, const xercesc::DOMElement *e=NULL) const;

    //! get the SWIG pointer of this value.
    void* getSwigThis(const Value &value) const;

    //! get the SWIG class name of this value.
    std::string getSwigType(const Value &value) const;

    Value createSwigByTypeName(const std::string &typeName) const;

    static octave_value_list fevalThrow(octave_function *func, const octave_value_list &arg, int n=0,
                                        const std::string &msg=std::string());

    virtual Value callFunction(const std::string &name, const std::vector<Value>& args) const;

    virtual double                            cast_double              (const Value &value) const;
    virtual std::vector<double>               cast_vector_double       (const Value &value) const;
    virtual std::vector<std::vector<double> > cast_vector_vector_double(const Value &value) const;
    virtual std::string                       cast_string              (const Value &value) const;

    virtual Value create_double              (const double& v) const;
    virtual Value create_vector_double       (const std::vector<double>& v) const;
    virtual Value create_vector_vector_double(const std::vector<std::vector<double> >& v) const;
    virtual Value create_string              (const std::string& v) const;
};

} // end namespace MBXMLUtils

#endif
