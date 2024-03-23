#ifndef _MBXMLUTILS_PREPROCESS_H_
#define _MBXMLUTILS_PREPROCESS_H_

#include <fmatvec/atom.h>
#include <mbxmlutilshelper/dom.h>
#include <mbxmlutils/eval.h>

namespace MBXMLUtils {

class Preprocess : virtual public fmatvec::Atom {
  public:
    using ParamSet = std::map<std::string, Eval::Value>;

    //! Instantiate a preprocessor using a inputFile and a parser.
    //! The inputFile is read and validated.
    Preprocess(const boost::filesystem::path &inputFile, // a filename of a XML file used as input
               std::variant<
                 std::shared_ptr<MBXMLUtils::DOMParser>, // a direct parser OR
                 xercesc::DOMElement*, // the root element of a DOM tree of a XML catalog file to create a parser OR
                 boost::filesystem::path // a filename of a XML catalog file to create a parser
               > parserVariant
              );
    //! Instantiate a preprocessor using a already parsed DOMDocument
    //! The inputDoc is validated.
    Preprocess(const std::shared_ptr<xercesc::DOMDocument> &inputDoc);

    //! Set top level parameters to overwrite before processAndGetDocument is called
    void setParam(const std::shared_ptr<ParamSet>& param_);
    //! Get available top level parameters after processAndGetDocument is called
    std::shared_ptr<ParamSet> getParam() const;

    //! Get the evaluator
    std::shared_ptr<Eval> getEvaluator() const;

    //! Get all dependencies found during processAndGetDocument
    const std::vector<boost::filesystem::path>& getDependencies() const;

    //! Start preprocessing and return the preprocessed DOMDocument
    std::shared_ptr<xercesc::DOMDocument> processAndGetDocument();

  private:
    std::shared_ptr<xercesc::DOMDocument> document;
    std::vector<boost::filesystem::path> dependencies;
    std::shared_ptr<Eval> eval;
    std::shared_ptr<ParamSet> param;

    bool preprocessed { false };

    void extractEvaluator();

    using PositionMap = std::map<FQN, int>;
    void preprocess(xercesc::DOMElement *&e, // in: element to process; out: e changes only if e is itself a Embed element
                    // in: root level parameters to overwrite; out: root level parameters
                    const std::shared_ptr<ParamSet>& param=std::shared_ptr<ParamSet>(),

                    // internal: XPath expression of parent element
                    const std::string &parentXPath="",
                    // internal: XPath expression of parent element
                    int embedXPathCount=1,
                    // internal: XPath position count of the element e
                    const std::shared_ptr<PositionMap>& position=std::make_shared<PositionMap>()
                   );
};

}

#endif
