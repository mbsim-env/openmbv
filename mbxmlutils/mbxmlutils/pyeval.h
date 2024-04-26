#ifndef _MBXMLUTILS_PYEVAL_H_
#define _MBXMLUTILS_PYEVAL_H_

#include "eval.h"
#include "pycppwrapper.h"

namespace MBXMLUtils {

/*! A evaluator based on python.
 *
 * See measurementToHtml.xsl for details.
 */
class PyEval : public Eval {
  friend class Eval;

  protected:
    //! Constructor.
    PyEval(std::vector<boost::filesystem::path> *dependencies_=nullptr);
  public:
    //! Destructor.
    ~PyEval() override;
    static std::string getNameStatic() { return "python"; }
    std::string getName() const override { return getNameStatic(); }
    void addImport(const std::string &code, const xercesc::DOMElement *e) override;
    bool valueIsOfType(const Value &value, ValueType type) const override;
    std::map<boost::filesystem::path, std::pair<boost::filesystem::path, bool> >& requiredFiles() const override;
    void convertIndex(Value &v, bool evalTo1Base) override;
    std::string getStringRepresentation(const Value &x) const override;
  protected:
    Value createFunctionIndep(int dim) const override;
    Value callFunction(const std::string &name, const std::vector<Value>& args) const override;
    Value fullStringToValue(const std::string &str, const xercesc::DOMElement *e, bool skipRet=false) const override;
  private:
    double                           cast_double                (const Value &value) const override;
    std::vector<double>              cast_vector_double         (const Value &value) const override;
    std::vector<std::vector<double> >cast_vector_vector_double  (const Value &value) const override;
    std::string                      cast_string                (const Value &value) const override;
    Value          create_double                   (const double& v) const override;
    Value          create_vector_double            (const std::vector<double>& v) const override;
    Value          create_vector_vector_double     (const std::vector<std::vector<double> >& v) const override;
    Value          create_string                   (const std::string& v) const override;

    Value          createFunctionDep(const std::vector<Value>& v) const override;
    Value          createFunctionDep(const std::vector<std::vector<Value> >& v) const override;
    Value          createFunction(const std::vector<Value> &indeps, const Value &dep) const override;

    std::string serializeFunction(const Value &x) const override;
    mutable std::map<size_t, PythonCpp::PyO> byteCodeMap;
};

}

#endif
