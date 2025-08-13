#ifndef _MBXMLUTILS_OCTEVAL_H_
#define _MBXMLUTILS_OCTEVAL_H_

#include "eval.h"
#include <memory>

class octave_value;
class octave_value_list;
class octave_function;

namespace fmatvec {
  class SymbolicExpression;
}

namespace XERCES_CPP_NAMESPACE { class DOMElement; }

namespace MBXMLUtils {

std::shared_ptr<octave_value> C(const Eval::Value &value);

Eval::Value C(const octave_value &value);

class OctEval;

/*! A evaluator based on octave.
 *
 * See measurementToHtml.xsl for details.
 */
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

    //! Add octave search path to the current evaluator context.
    //! \p code must evaluate to a string representing a directory/path.
    //! A relative path is expanded to an absolute path using the path of e as current directory.
    //! The absolute path is then added using "addpath" to the octave search path.
    void addImport(const std::string &code, const xercesc::DOMElement *e, const std::string &action="") override;

    void addImportHelper(const boost::filesystem::path &dir);

    //! get the type of value
    bool valueIsOfType(const Value &value, ValueType type) const override;

    //! return a list of all required files of octave (excluding dependent files of libraries)
    std::map<boost::filesystem::path, std::pair<boost::filesystem::path, bool> >& requiredFiles() const override;

    void convertIndex(Value &v, bool evalTo1Based) override {}

  protected:

    struct Import {
      std::string path;
      std::map<std::string, octave_value> vn;
      std::map<std::string, octave_value> gvn;
      std::map<std::string, octave_value> ufn;
      std::map<std::string, octave_value> tlvn;
    };

    Value createFunctionIndep(int dim) const override;

    //! evaluate str fully and return result as an octave variable
    Value fullStringToValue(const std::string &str, const xercesc::DOMElement *e, bool skipRet=false) const override;

    static octave_value_list fevalThrow(octave_function *func, const octave_value_list &arg, int n=0,
                                        const std::string &msg=std::string());

    static void* getSwigPtr(const octave_value &v);
    static Value createSwigByTypeName(const std::string &name);
    static std::string getSwigType(const octave_value &value);

    Value callFunction(const std::string &name, const std::vector<Value>& args) const override;

    double                            cast_double              (const Value &value) const override;
    std::vector<double>               cast_vector_double       (const Value &value) const override;
    std::vector<std::vector<double> > cast_vector_vector_double(const Value &value) const override;
    std::string                       cast_string              (const Value &value) const override;

    Value create_double              (const double& v) const override;
    Value create_vector_double       (const std::vector<double>& v) const override;
    Value create_vector_vector_double(const std::vector<std::vector<double> >& v) const override;
    Value create_string              (const std::string& v) const override;

    std::string createSourceCode_double              (const double& v) const override;
    std::string createSourceCode_vector_double       (const std::vector<double>& v) const override;
    std::string createSourceCode_vector_vector_double(const std::vector<std::vector<double> >& v) const override;
    std::string createSourceCode_string              (const std::string& v) const override;

    Value createFunctionDep(const std::vector<Value>& v) const override;
    Value createFunctionDep(const std::vector<std::vector<Value> >& v) const override;
    Value createFunction(const std::vector<Value> &indeps, const Value &dep) const override;

    std::string serializeFunction(const Value &x) const override;
};

} // end namespace MBXMLUtils

#endif
