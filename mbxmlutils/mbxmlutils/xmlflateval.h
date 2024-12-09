#ifndef _MBXMLUTILS_XMLFLATEVAL_H_
#define _MBXMLUTILS_XMLFLATEVAL_H_

#include "eval.h"

namespace MBXMLUtils {

/*! A (dummy) evaluator taking the xmlflat syntax as input and output.
 * This evaluator cannot evaluate any expresion nor any parametrisation.
 * Its just a dummy evaluator for testing. But it can be used for evaluate xmlflat file or a normal file
 * without and parametrisation and without any expression. */
class XMLFlatEval : public Eval {
  friend class Eval;
  protected:
    XMLFlatEval(std::vector<boost::filesystem::path> *dependencies_=nullptr);
  public:
    ~XMLFlatEval() override;
    static std::string getNameStatic() { return "xmlflat"; }
    std::string getName() const override { return getNameStatic(); }
    void addImport(const std::string &code, const xercesc::DOMElement *e, const std::string &action="") override;
    bool valueIsOfType(const Value &value, ValueType type) const override;
    std::map<boost::filesystem::path, std::pair<boost::filesystem::path, bool> >& requiredFiles() const override;
    void convertIndex(Value &v, bool evalTo1Based) override {}
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
};

} // end namespace MBXMLUtils

#endif
