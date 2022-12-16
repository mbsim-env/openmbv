#ifndef _MBXMLUTILS_PREPROCESS_H_
#define _MBXMLUTILS_PREPROCESS_H_

#include <fmatvec/atom.h>
#include <mbxmlutilshelper/dom.h>
#include <mbxmlutils/eval.h>

namespace MBXMLUtils {

class Preprocess : virtual public fmatvec::Atom {
  protected:
    using PositionMap = std::map<FQN, int>;
  public:
    using ParamSet = std::map<std::string, Eval::Value>;
    static void preprocess(const std::shared_ptr<MBXMLUtils::DOMParser>& parser, // in: parser used to parse XML documents
                           const std::shared_ptr<Eval> &eval, // in: evaluator used for evaluation
                           std::vector<boost::filesystem::path> &dependencies, // out: list of dependent files
                           xercesc::DOMElement *&e, // in: element to process; out: e changes only if e is itself a Embed element
                           // in: root level parameters to overwrite; out: root level parameters
                           const std::shared_ptr<ParamSet>& param=std::shared_ptr<ParamSet>(),

                           // internal: XPath expression of parent element
                           const std::string &parentXPath="",
                           // internal: XPath expression of parent element
                           int embedXPathCount=1,
                           // internal: XPath position count of the element e
                           const std::shared_ptr<PositionMap>& position=std::make_shared<PositionMap>()
                          );

    // same as process but reads from mainXML and return preprocessed DOMDocument.
    static std::shared_ptr<xercesc::DOMDocument> preprocessFile(
      std::vector<boost::filesystem::path> &dependencies, std::set<boost::filesystem::path> schemas,
      const boost::filesystem::path &mainXML);
};

}

#endif
