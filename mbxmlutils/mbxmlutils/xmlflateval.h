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
    virtual std::string getName() { return getNameStatic(); }
    virtual void addPath(const boost::filesystem::path &dir, const xercesc::DOMElement *e);
    virtual bool valueIsOfType(const boost::shared_ptr<void> &value, ValueType type);
    virtual std::map<boost::filesystem::path, std::pair<boost::filesystem::path, bool> >& requiredFiles();
  protected:
    virtual boost::shared_ptr<void> createSwigByTypeName(const std::string &typeName);
    virtual boost::shared_ptr<void> callFunction(const std::string &name, const std::vector<boost::shared_ptr<void> >& args);
    virtual boost::shared_ptr<void> fullStringToValue(const std::string &str, const xercesc::DOMElement *e=NULL);
    virtual void* getSwigThis(const boost::shared_ptr<void> &value);
    virtual std::string getSwigType(const boost::shared_ptr<void> &value);
  private:
    virtual double                           cast_double                (const boost::shared_ptr<void> &value);
    virtual std::vector<double>              cast_vector_double         (const boost::shared_ptr<void> &value);
    virtual std::vector<std::vector<double> >cast_vector_vector_double  (const boost::shared_ptr<void> &value);
    virtual std::string                      cast_string                (const boost::shared_ptr<void> &value);
    virtual boost::shared_ptr<void>          create_double              (const double& v);
    virtual boost::shared_ptr<void>          create_vector_double       (const std::vector<double>& v);
    virtual boost::shared_ptr<void>          create_vector_vector_double(const std::vector<std::vector<double> >& v);
    virtual boost::shared_ptr<void>          create_string              (const std::string& v);
};

} // end namespace MBXMLUtils

#endif
