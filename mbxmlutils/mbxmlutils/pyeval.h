#ifndef _MBXMLUTILS_PYEVAL_H_
#define _MBXMLUTILS_PYEVAL_H_

#include "eval.h"
#include "py2py3cppwrapper.h"

namespace MBXMLUtils {

//! A evaluator based on python.
class PyEval : public Eval {
  friend class Eval;

  protected:
    //! Constructor.
    PyEval(std::vector<boost::filesystem::path> *dependencies_=NULL);
  public:
    //! Destructor.
    ~PyEval();
    static std::string getNameStatic() { return "python"; }
    virtual std::string getName() const { return getNameStatic(); }
    virtual void addImport(const std::string &code, const xercesc::DOMElement *e, bool deprecated=false);
    virtual bool valueIsOfType(const Value &value, ValueType type) const;
    virtual std::map<boost::filesystem::path, std::pair<boost::filesystem::path, bool> >& requiredFiles() const;
    virtual bool useOneBasedIndexes() { return false; }
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

}

#endif
