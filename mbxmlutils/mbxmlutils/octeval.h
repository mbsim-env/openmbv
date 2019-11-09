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
    OctEval(std::vector<boost::filesystem::path> *dependencies_=nullptr);

  public:
    //! Destructor.
    ~OctEval() override;

    //! Get the name of this evaluator.
    static std::string getNameStatic() { return "octave"; }
    std::string getName() const override { return getNameStatic(); }

    std::shared_ptr<void> addIndependentVariableParam(const std::string &paramName, int dim) override;

    //! Add octave search path to the current evaluator context.
    //! \p code must evaluate to a string representing a directory/path.
    //! A relative path is expanded to an absolute path using the path of e as current directory.
    //! The absolute path is then added using "addpath" to the octave search path.
    void addImport(const std::string &code, const xercesc::DOMElement *e) override;

    //! get the type of value
    bool valueIsOfType(const Value &value, ValueType type) const override;

    //! return a list of all required files of octave (excluding dependent files of libraries)
    std::map<boost::filesystem::path, std::pair<boost::filesystem::path, bool> >& requiredFiles() const override;

    void convertIndex(Value &v, bool evalTo1Based) override {}

  protected:

    //! evaluate str fully and return result as an octave variable
    Value fullStringToValue(const std::string &str, const xercesc::DOMElement *e) const override;

    static octave_value_list fevalThrow(octave_function *func, const octave_value_list &arg, int n=0,
                                        const std::string &msg=std::string());

    Value callFunction(const std::string &name, const std::vector<Value>& args) const override;

    double                            cast_double              (const Value &value) const override;
    std::vector<double>               cast_vector_double       (const Value &value) const override;
    std::vector<std::vector<double> > cast_vector_vector_double(const Value &value) const override;
    std::string                       cast_string              (const Value &value) const override;
    Function                          cast_Function            (const Value &value) const override;

    Value create_double              (const double& v) const override;
    Value create_vector_double       (const std::vector<double>& v) const override;
    Value create_vector_vector_double(const std::vector<std::vector<double> >& v) const override;
    Value create_string              (const std::string& v) const override;
    Value create_vector_void         (const std::vector<std::shared_ptr<void>>& v) const override;
    Value create_vector_vector_void  (const std::vector<std::vector<std::shared_ptr<void>> >& v) const override;

    std::string serializeFunction(const std::shared_ptr<void> &x) const override;
};

} // end namespace MBXMLUtils

#endif
