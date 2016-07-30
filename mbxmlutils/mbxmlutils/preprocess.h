#ifndef _MBXMLUTILS_PREPROCESS_H_
#define _MBXMLUTILS_PREPROCESS_H_

#include <fmatvec/atom.h>
#include <mbxmlutilshelper/dom.h>
#include <mbxmlutils/eval.h>

namespace MBXMLUtils {


class Preprocess : virtual public fmatvec::Atom {
  protected:
    typedef std::map<FQN, int> PositionMap;
  public:
    typedef std::vector<std::pair<std::string, std::shared_ptr<void> > > ParamSet;
    typedef std::unordered_map<std::string, ParamSet> XPathParamSet;
    static void preprocess(std::shared_ptr<MBXMLUtils::DOMParser> parser, // in: parser used to parse XML documents
                           const std::shared_ptr<Eval> &eval, // in: evaluator used for evaluation
                           std::vector<boost::filesystem::path> &dependencies, // out: list of dependent files
                           xercesc::DOMElement *&e, // in: element to process; out: e changes only if e is itself a Embed element
                           // out: XPath map of top level parameter sets. Note: the XPath position is always interpreted
                           //      with a Embed count of 1!
                           std::shared_ptr<XPathParamSet> param=std::shared_ptr<XPathParamSet>(),

                           // internal: XPath expression of parent element
                           const std::string &parentXPath="",
                           // internal: XPath position count of the element e
                           std::shared_ptr<PositionMap> position=std::make_shared<PositionMap>()
                          );
};

}

#endif
