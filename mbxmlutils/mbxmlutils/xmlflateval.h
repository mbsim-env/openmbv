#ifndef _MBXMLUTILS_XMLFLATEVAL_H_
#define _MBXMLUTILS_XMLFLATEVAL_H_

#include "eval.h"

namespace MBXMLUtils {

/*! A (dummy) evaluator taking the xmlflat syntax as input and output.
 * This evaluator cannot evaluate any expresion nor any parametrisation or casadi actions can be done.
 * Its just a dummy evaluator for testing. But it can be used for evaluate xmlflat file or a normal file
 * without and parametrisation and without any expression. */
class XMLFlatEval : public Eval {
  friend class Eval;
  protected:
    XMLFlatEval(std::vector<boost::filesystem::path> *dependencies_=NULL);
  public:
    ~XMLFlatEval();
    static std::string getNameStatic() { return "xmlflat"; }
    virtual std::string getName() const { return getNameStatic(); }
    virtual void addImport(const std::string &code, const xercesc::DOMElement *e);
    virtual bool valueIsOfType(const Value &value, ValueType type) const;
    virtual std::map<boost::filesystem::path, std::pair<boost::filesystem::path, bool> >& requiredFiles() const;
    virtual bool useOneBasedIndexes() { return true; }
  protected:
    virtual Value createSwigByTypeName(const std::string &typeName) const;
    virtual Value callFunction(const std::string &name, const std::vector<Value>& args) const;
    virtual Value fullStringToValue(const std::string &str, const xercesc::DOMElement *e=NULL) const;
    virtual void* getSwigThis(const Value &value) const;
    virtual std::string getSwigType(const Value &value) const;
  private:
    virtual double                           cast_double                (const Value &value) const;
    virtual std::vector<double>              cast_vector_double         (const Value &value) const;
    virtual std::vector<std::vector<double> >cast_vector_vector_double  (const Value &value) const;
    virtual std::string                      cast_string                (const Value &value) const;
    virtual Value          create_double              (const double& v) const;
    virtual Value          create_vector_double       (const std::vector<double>& v) const;
    virtual Value          create_vector_vector_double(const std::vector<std::vector<double> >& v) const;
    virtual Value          create_string              (const std::string& v) const;
};

} // end namespace MBXMLUtils

#endif
