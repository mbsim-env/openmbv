#ifndef _MBXMLUTILSHELPER_DOM_H_
#define _MBXMLUTILSHELPER_DOM_H_

#include <fmatvec/atom.h>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <set>
#include <boost/filesystem.hpp>
#include <xercesc/dom/DOMErrorHandler.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMText.hpp>
#include <xercesc/dom/DOMLSParser.hpp>
#include <xercesc/dom/DOMLocator.hpp>
#include <xercesc/dom/DOMUserDataHandler.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/util/TransService.hpp>
#include <xercesc/util/XMLEntityResolver.hpp>
#include <xercesc/framework/psvi/PSVIHandler.hpp>
#include <boost/lexical_cast.hpp>
#include <fmatvec/toString.h>

namespace XERCES_CPP_NAMESPACE {
  class DOMProcessingInstruction;
  class DOMImplementation;
  class AbstractDOMParser;
}

namespace boost {
  template<> std::vector<double> lexical_cast(const std::string& str);
  template<> std::vector<std::vector<double>> lexical_cast(const std::string& str);
  template<> std::vector<int> lexical_cast(const std::string& str);
  template<> std::vector<std::vector<int>> lexical_cast(const std::string& str);
  template<> bool lexical_cast<bool>(const std::string& arg);
}

namespace {

template<class T>
struct CheckSize {
  static void check(const xercesc::DOMElement *me, const T &value, int r, int c);
};
template<class T>
struct CheckSize<std::vector<T>> {
  static void check(const xercesc::DOMElement *me, const std::vector<T> &value, int r, int c);
};
template<class T>
struct CheckSize<std::vector<std::vector<T>>> {
  static void check(const xercesc::DOMElement *me, const std::vector<std::vector<T>> &value, int r, int c);
};

}

namespace MBXMLUtils {

// forward declaration
template<typename DOMDocumentType>
class DOMDocumentWrapper;
template<typename DOMDocumentType>
DOMDocumentWrapper<DOMDocumentType> D(DOMDocumentType *me);
template<typename DOMElementType>
class DOMElementWrapper;
template<typename DOMElementType>
DOMElementWrapper<DOMElementType> E(DOMElementType *me);

//! Initialize Xerces on load and terminate it on unload of the library/program
class InitXerces {
  public:
    InitXerces() {
      xercesc::XMLPlatformUtils::Initialize();
    }
    ~InitXerces() {
      xercesc::XMLPlatformUtils::Terminate();
    }
};

//! Helper class to convert std::string (UTF8) to XMLCh* (UTF16) and the other way around.
//! A returned XMLCh pointer has a lifetime of the X object.
class X {
  public:
    const XMLCh *operator%(const std::string &str) {
      if(str.empty())
        return &xercesc::chNull;
      const XMLCh *unicode=xercesc::TranscodeFromStr(reinterpret_cast<const XMLByte*>(str.c_str()), str.length(), "UTF8").adopt();
      store.push_back(std::shared_ptr<const XMLCh>(unicode, &releaseXMLCh));
      return unicode;
    }
    const XMLCh *operator()(const std::string &str) { return operator%(str); }
    std::string operator%(const XMLCh *unicode) {
      if(!unicode || unicode[0]==0)
        return std::string();
      return reinterpret_cast<const char*>(xercesc::TranscodeToStr(unicode, "UTF8").str());
    }
    std::string operator()(const XMLCh *unicode) { return operator%(unicode); }
    static void releaseXMLCh(const XMLCh *s) { xercesc::XMLPlatformUtils::fgMemoryManager->deallocate(const_cast<XMLCh*>(s)); }
  private:
    std::vector<std::shared_ptr<const XMLCh> > store;
};

//! Print DOM error messages
class DOMErrorPrinter: public xercesc::DOMErrorHandler, virtual public fmatvec::Atom
{
  public:
    DOMErrorPrinter()  = default;
    bool handleError(const xercesc::DOMError&) override;
    int getNumWarnings() const { return warningCount; }
    int getNumErrors() const { return errorCount; }
    void resetCounter() { warningCount=0; errorCount=0; }
  protected:
    int warningCount{0};
    int errorCount{0};
};

//! Full qualified name.
//! A std::pair representing a full qualified XML name as a pair of namespace URI and local name.
class FQN : public std::pair<std::string, std::string> {
  public:
    //! Empty FQN
    FQN()  {}
    //! Anonymous FQN
    FQN(const std::string &name) : std::pair<std::string, std::string>("", name) {}
    //! Anonymous FQN (required for implicit casting of string literals to anonymous FQNs)
    FQN(const char *name) : std::pair<std::string, std::string>("", name) {}
    //! FQN
    FQN(const std::string &ns, const std::string &name) : std::pair<std::string, std::string>(ns, name) {}
};

//! Helper class to easily construct full qualified XML names (FQN) using XML namespace prefixes.
class NamespaceURI {
  public:
    NamespaceURI(std::string nsuri_) : nsuri(std::move(nsuri_)) {}
    FQN operator%(const std::string &localName) const { return FQN(nsuri, localName); }
    std::string getNamespaceURI() const { return nsuri; }
  private:
    std::string nsuri;
};

//! Declaration of the XML xinclude prefix/URI.
const NamespaceURI XINCLUDE("http://www.w3.org/2001/XInclude");
//! Declaration of the xmlns prefix/URI.
const NamespaceURI XMLNS("http://www.w3.org/2000/xmlns/");
//! Declaration of the MBXMLUtils physicalvariable namespace prefix/URI.
const NamespaceURI PV("http://www.mbsim-env.de/MBXMLUtils");

//! Helper class for DOMEvalException.
//! Extension of DOMLocator with a embed count
class EmbedDOMLocator : public xercesc::DOMLocator {
  public:
    EmbedDOMLocator(const boost::filesystem::path &file_, int row_, int embedCount_, std::string xpath_) : DOMLocator(),
      file(x%file_.string()), row(row_), embedCount(embedCount_), xpath(std::move(xpath_)) {}
    EmbedDOMLocator(const EmbedDOMLocator &src) : DOMLocator(),
      file(x%(X()%src.file)), row(src.row), embedCount(src.embedCount), xpath(src.xpath) {}
    EmbedDOMLocator& operator=(const EmbedDOMLocator &src) {
      file=x%(X()%src.file);
      row=src.row;
      embedCount=src.embedCount;
      xpath=src.xpath;
      return *this;
    }
    ~EmbedDOMLocator() = default;
    EmbedDOMLocator(const EmbedDOMLocator &&src) = delete;
    EmbedDOMLocator& operator=(const EmbedDOMLocator &&src) = delete;
    XMLFileLoc getLineNumber() const override { return row; }
    XMLFileLoc getColumnNumber() const override { return 0; }
    XMLFilePos getByteOffset() const override { return ~(XMLFilePos(0)); }
    XMLFilePos getUtf16Offset() const override { return ~(XMLFilePos(0)); }
    xercesc::DOMNode *getRelatedNode() const override { return nullptr; }
    const XMLCh *getURI() const override { return file; }
    int getEmbedCount() const { return embedCount; }
    std::string getRootXPathExpression() const { return xpath; }
  private:
    X x;
    const XMLCh *file;
    int row;
    int embedCount;
    std::string xpath;
};

// Exception wrapping for DOMEvalException.
// Catch a exception of type DOMEvalException, set the current XML context to DOMElement e and rethrow.
// Not not change the context if it was alread set before.
#define RETHROW_MBXMLUTILS(e) \
  catch(MBXMLUtils::DOMEvalException &ex) { \
    if(ex.getLocationStack().size()==0) \
      ex.setContext(e); \
    throw; \
  } \
  catch(const std::exception &ex) { \
    throw DOMEvalException(ex.what(), e); \
  }

//! Exception during evaluation of the DOM tree including a location stack
class DOMEvalException : public std::exception {
  public:
    DOMEvalException(const std::string &errorMsg_, const xercesc::DOMElement *e=nullptr, const xercesc::DOMAttr *a=nullptr);
    DOMEvalException(const DOMEvalException &src) : errorMsg(src.errorMsg),
      locationStack(src.locationStack) {}
    DOMEvalException &operator=(const DOMEvalException &src) {
      errorMsg=src.errorMsg; locationStack=src.locationStack;
      return *this;
    }
    ~DOMEvalException() noexcept override = default;
    static void generateLocationStack(const xercesc::DOMElement *e, const xercesc::DOMAttr *a,
                                      std::vector<EmbedDOMLocator> &locationStack);
    static std::string errorLocationOutput(const std::string &indent, const std::vector<EmbedDOMLocator> &locationStack,
                                           const std::string &message="");

    /** Convert a error location and error message for outputting it to the console.
      The behaviour of this function can be adapted by the environment variable MBXMLUTILS_ERROROUTPUT.
      For each error the value of this variable is printed, where the value is interpreted as a
      "Boost-Extended Format String Syntax", where the following named sub-expressions are recognized:
     
      named sub-expr  | value of the named sub-expression
      --------------- | -------------------------------------------
      msg             | the error message
      file            | the filename where the error occured
      line            | the line number in the file where the error occured
      xpath           | the XPath expression from the root element of the file to the element in the file where the error occured
      ecount          | the embed count number where the error occured
      sse             | undefined value but only defined if this is a subsequent error
     
      All these named sub-expressions may not be defined (see "Boost-Extended Format Syntax Syntax" on how to handle this).
      If the environment variable MBXMLUTILS_ERROROUTPUT is not set then the following is used as default:
      \verbatim
      $+{file}(?{line}\:$+{line}:)(?{ecount}[count=$+{ecount}]:)(?{msg}\: $+{msg}:)
      \endverbatim
     
      Beside the this default value also the following special values can be used for MBXMLUTILS_ERROROUTPUT:
     
      GCC: is equal to the default value above
     
      HTML: uses the following:
      \verbatim
      <a href="$+{file}(?{line}\?line=$+{line}:)">$+{file}(?{line}\:$+{line}:)</a>(?{ecount}[count=$+{ecount}]:)(?{msg}\: $+{msg}:)
      \endverbatim
     
      XPATH: uses the following:
      \verbatim
      <error file="$+{file}" xpath="$+{xpath}"(?{ecount} ecount="$+{ecount}":) sse="(?{sse}1:0)">$+{msg}</error>
      \endverbatim
     */
    static std::string errorOutput(const xercesc::DOMLocator &loc, const std::string &message, bool subsequentError=false);

    void setContext(const xercesc::DOMElement *e, const xercesc::DOMAttr* a=nullptr);
    const std::string& getMessage() const { return errorMsg; }
    const std::vector<EmbedDOMLocator>& getLocationStack() const { return locationStack; }
    const char* what() const noexcept override;
  private:
    std::string errorMsg;
    std::vector<EmbedDOMLocator> locationStack;
    std::string attrName;
    mutable std::string whatStr;
};

//! Helper class for extending DOMElement (use the function E(...)).
template<typename DOMElementType>
class DOMElementWrapper {
  public:
    //! Wrap DOMElement to my special element
    DOMElementWrapper(DOMElementType *me_) : me(me_) {}
    //! Get full qualified tag name
    FQN getTagName() const { return FQN(X()%me->getNamespaceURI(), X()%me->getLocalName()); }
    //! Get first child element of the specified full qualified name
    const xercesc::DOMElement *getFirstElementChildNamed(const FQN &name) const;
    //! Get first child element of the specified full qualified name
    xercesc::DOMElement *getFirstElementChildNamed(const FQN &name);
    //! Get next sibling element of the specified full qualified name
    const xercesc::DOMElement *getNextElementSiblingNamed(const FQN &name) const;
    //! Get next sibling element of the specified full qualified name
    xercesc::DOMElement *getNextElementSiblingNamed(const FQN &name);
    //! Get first child processing instruction of the specified target
    const xercesc::DOMProcessingInstruction *getFirstProcessingInstructionChildNamed(const std::string &target) const;
    //! Get first child processing instruction of the specified target
    xercesc::DOMProcessingInstruction *getFirstProcessingInstructionChildNamed(const std::string &target);
    //! Get first child text
    const xercesc::DOMText *getFirstTextChild() const;
    //! Get first child text
    xercesc::DOMText *getFirstTextChild();
    //! Get the child text as type T
    template<class T> T getText(int r=0, int c=0) const {
      try {
        auto ret=boost::lexical_cast<T>(X()%E(me)->getFirstTextChild()->getData());
        CheckSize<T>::check(me, ret, r, c);
        return ret;
      }
      catch(const boost::bad_lexical_cast &ex) {
        throw DOMEvalException(ex.what(), me);
      }
      catch(const std::exception &ex) {
        throw DOMEvalException(ex.what(), me);
      }
    }

    template<class T> void addElementText(const FQN &name, const T &value) {
      xercesc::DOMElement *ele=D(me->getOwnerDocument())->createElement(name);
      ele->insertBefore(me->getOwnerDocument()->createTextNode(MBXMLUtils::X()%fmatvec::toString(value)), nullptr);
      me->insertBefore(ele, nullptr);
    }
    //! Check if the element is of type base
    //! Note DOMTypeInfo::isDerivedFrom is not implemented in xerces-c hence we define our one methode here.
    bool isDerivedFrom(const FQN &baseTypeName) const;
    //! Get the base URI.
    //! Returns the value of the first OriginalFileName PI define by a parent element or the document uri if no such element is found.
    //! If skipThis is false the OriginalFileName PI of this element in honored else
    //! this element is skipped and only parent elements are honored.
    //! If a OriginalFileName PI was found 'found' is set to this element else 'found' is NULL.
    boost::filesystem::path getOriginalFilename(bool skipThis=false,
                                                const xercesc::DOMElement *&found=DOMElementWrapper<DOMElementType>::dummyArg) const;
    //! Set original filename.
    //! Calls getOriginalFilename on itself and set this value to itself.
    //! This function should/must be called when a element is removed from a tree but still used after that.
    void setOriginalFilename();
    //! Convert the relative path relPath to an aboslute path by prefixing it with the path of this document.
    //! If relPath is a absolute path it is returned as it. (see also getOriginalFilename)
    boost::filesystem::path convertPath(const boost::filesystem::path &relPath) const;
    //! Get the line number.
    //! If a LineNr processing instruction child node exist this number is returned. If not the XML line number is returned.
    int getLineNumber() const;
    //! Get the embed count.
    //! If a EmbedCount processing instruction child node exist this number is returned. If not 0 is returned.
    int getEmbedCountNumber() const;
    //! Set the embed count.
    //! Is store as a processing instruction child node.
    void setEmbedCountNumber(int embedCount);
    //! Get the embed XPath count.
    //! If a EmbedXPathCount processing instruction child node exist this number is returned. If not 0 is returned.
    int getEmbedXPathCount() const;
    //! Set the embed XPath count.
    //! Is store as a processing instruction child node.
    void setEmbedXPathCount(int xPathCount);
    //! Get the XPath from the root element to this element.
    //! The root element may not be the document itself if embedding has occured.
    std::string getRootXPathExpression() const;
    //! Get the line number of the original element
    int getOriginalElementLineNumber() const;
    //! Set the line number of the original element
    void setOriginalElementLineNumber(int lineNr);
    //! Get attribute named name.
    std::string getAttribute(const FQN &name) const;
    //! Get attribute named name of type QName.
    FQN getAttributeQName(const FQN &name) const;
    //! Get attribute node named name.
    const xercesc::DOMAttr* getAttributeNode(const FQN &name) const;
    //! Get attribute node named name.
    xercesc::DOMAttr* getAttributeNode(const FQN &name);
    //! Set attribute.
    template<class T>
    void setAttribute(const FQN &name, const T &value) {
      me->setAttributeNS(X()%name.first, X()%name.second, X()%fmatvec::toString(value));
    }
    //! Set attribute of type FQN.
    void setAttribute(const FQN &name, const FQN &value);
    //! check if this element has a attibute named name.
    bool hasAttribute(const FQN &name) const;
    //! remove from this element the attibute named name.
    void removeAttribute(const FQN &name);
    //! Workaround: convert default attributes to normal attributes (must be used before importNode to also import default attributes)
    void workaroundDefaultAttributesOnImportNode();
    //! Treat this object as a pointer (like DOMElement*)
    typename std::conditional<std::is_same<DOMElementType, const xercesc::DOMElement>::value,
      const DOMElementWrapper*, DOMElementWrapper*>::type operator->() {
      return this;
    }
    static const xercesc::DOMElement *dummyArg;
  private:
    DOMElementType *me;
};
//! Helper function, with a very short name, for automatic type deduction for DOMElementWrapper.
template<typename DOMElementType>
DOMElementWrapper<DOMElementType> E(DOMElementType *me) { return DOMElementWrapper<DOMElementType>(me); }
//! Helper function, with a very short name, for automatic type deduction for DOMElementWrapper.
template<typename DOMElementType>
DOMElementWrapper<DOMElementType> E(std::shared_ptr<DOMElementType> me) { return DOMElementWrapper<DOMElementType>(me.get()); }

//! Helper class for extending DOMAttr (use the function A(...)).
template<typename DOMAttrType>
class DOMAttrWrapper {
  public:
    //! Wrap DOMAttr to my special element
    DOMAttrWrapper(DOMAttrType *me_) : me(me_) {}
    //! Check if the element is of type base
    //! Note DOMTypeInfo::isDerivedFrom is not implemented in xerces-c hence we define our one methode here.
    bool isDerivedFrom(const FQN &baseTypeName) const;
    //! Get the XPath from the root element to this attribute.
    //! The root element may not be the document itself if embedding has occured.
    std::string getRootXPathExpression() const;
    //! Treat this object as a pointer (like DOMAttr*)
    typename std::conditional<std::is_same<DOMAttrType, const xercesc::DOMAttr>::value,
      const DOMAttrWrapper*, DOMAttrWrapper*>::type operator->() {
      return this;
    }
  private:
    DOMAttrType *me;
};
//! Helper function, with a very short name, for automatic type deduction for DOMAttrWrapper.
template<typename DOMAttrType>
DOMAttrWrapper<DOMAttrType> A(DOMAttrType *me) { return DOMAttrWrapper<DOMAttrType>(me); }
//! Helper function, with a very short name, for automatic type deduction for DOMAttrWrapper.
template<typename DOMAttrType>
DOMAttrWrapper<DOMAttrType> A(std::shared_ptr<DOMAttrType> me) { return DOMAttrWrapper<DOMAttrType>(me.get()); }

class DOMParser;

//! Helper class for extending DOMDocument (use the function D(...)).
template<typename DOMDocumentType>
class DOMDocumentWrapper {
  public:
    //! Wrap DOMDocument to my special element
    DOMDocumentWrapper(DOMDocumentType *me_) : me(me_) {}
    //! (re)validate the document using the parser this document was created
    void validate();
    //! create element with the given FQN
    //! Note: a empty namespace (name.first.empty()==true) as no namespace
    xercesc::DOMElement* createElement(const FQN &name);
    //! Get full qualified tag name
    std::shared_ptr<DOMParser> getParser() const;
    //! Get the filename of the document.
    //! This is the same as getDocumentURI but with the file schema removed.
    boost::filesystem::path getDocumentFilename() const;
    //! Get the node (DOMElement or DOMAttrType) corresponding the given xpathExpression relative to the root.
    //! If context is nullptr than the root element is used.
    //! Only a very small subset of XPath is supported by this function (just the one returned by getRootXPathExpression)
    xercesc::DOMNode* evalRootXPathExpression(std::string xpathExpression, xercesc::DOMElement* context=nullptr);
    //! Treat this object as a pointer (like DOMDocument*)
    typename std::conditional<std::is_same<DOMDocumentType, const xercesc::DOMDocument>::value,
      const DOMDocumentWrapper*, DOMDocumentWrapper*>::type operator->() {
      return this;
    }
  private:
    DOMDocumentType *me;
};
//! Helper function, with a very short name, for automatic type deduction for DOMDocumentWrapper.
template<typename DOMDocumentType>
DOMDocumentWrapper<DOMDocumentType> D(DOMDocumentType *me) { return DOMDocumentWrapper<DOMDocumentType>(me); }
//! Helper function, with a very short name, for automatic type deduction for DOMDocumentWrapper.
template<typename DOMDocumentType>
DOMDocumentWrapper<DOMDocumentType> D(std::shared_ptr<DOMDocumentType> me) { return DOMDocumentWrapper<DOMDocumentType>(me.get()); }

class LocationInfoFilter : public xercesc::DOMLSParserFilter {
  public:
    void setParser(DOMParser *parser_) { parser=parser_; }
    xercesc::DOMLSParserFilter::FilterAction acceptNode(xercesc::DOMNode *n) override;
    xercesc::DOMLSParserFilter::FilterAction startElement(xercesc::DOMElement *e) override;
    xercesc::DOMNodeFilter::ShowType getWhatToShow() const override;
  private:
    DOMParser *parser;
    static const std::string lineNumberKey;
};

class TypeDerivativeHandler : public xercesc::PSVIHandler, virtual public fmatvec::Atom {
  public:
    void setParser(DOMParser *parser_) { parser=parser_; }
    void handleElementPSVI(const XMLCh *localName, const XMLCh *uri, xercesc::PSVIElement *info) override;
    void handleAttributesPSVI(const XMLCh *localName, const XMLCh *uri, xercesc::PSVIAttributeList *psviAttributes) override;
  private:
    DOMParser *parser;
};

class DOMParserUserDataHandler : public xercesc::DOMUserDataHandler {
  public:
    void handle(DOMOperationType operation, const XMLCh* key, void *data, const xercesc::DOMNode *src, xercesc::DOMNode *dst) override;
};

class EntityResolver : public xercesc::XMLEntityResolver {
  public:
    void setParser(DOMParser *parser_) { parser=parser_; }
    xercesc::InputSource* resolveEntity(xercesc::XMLResourceIdentifier *resourceIdentifier) override;
  private:
    DOMParser *parser;
};

//! A XML DOM parser.
class DOMParser : public std::enable_shared_from_this<DOMParser> {
  friend bool isDerivedFrom(const xercesc::DOMNode *me, const FQN &baseTypeName);
  friend class TypeDerivativeHandler;
  friend class LocationInfoFilter;
  friend class DOMParserUserDataHandler;
  friend class EntityResolver;
  template<typename> friend class DOMDocumentWrapper;
  public:
    //! Create DOM parser
    static std::shared_ptr<DOMParser> create(const std::set<boost::filesystem::path> &schemas={});
    //! Parse a XML document
    std::shared_ptr<xercesc::DOMDocument> parse(const boost::filesystem::path &inputSource,
                                                  std::vector<boost::filesystem::path> *dependencies=nullptr);
    //! Serialize a document to a file.
    //! Helper function to write a node. This normalized the document before.
    static void serialize(xercesc::DOMNode *n, const boost::filesystem::path &outputSource, bool prettyPrint=true);
    //! Serialize a document to a memory (std::string).
    //! Helper function to write a node. This normalized the document before.
    static void serializeToString(xercesc::DOMNode *n, std::string &outputData, bool prettyPrint=true);
    //! reset all loaded grammars
    void resetCachedGrammarPool();
    //! create a empty document
    std::shared_ptr<xercesc::DOMDocument> createDocument();
  private:
    static const std::string domParserKey;

    // Load XML Schema grammar file: is actually loaded (the main XML Schema file of a XML file must be loaded direclty)
    void loadGrammar(const boost::filesystem::path &schemaFilename);
    // Register XML Schema grammar file: is loaded on demand if needed by another XML Schema
    void registerGrammar(const std::shared_ptr<DOMParser> &nonValParser, const boost::filesystem::path &schemaFilename);

    xercesc::DOMImplementation *domImpl;
    DOMParser(const std::set<boost::filesystem::path> &schemas);
    std::shared_ptr<xercesc::DOMLSParser> parser;
    std::map<FQN, xercesc::XSTypeDefinition*> typeMap;
    DOMErrorPrinter errorHandler;
    LocationInfoFilter locationFilter;
    TypeDerivativeHandler typeDerHandler;
    EntityResolver entityResolver;
    static DOMParserUserDataHandler userDataHandler;
    std::map<std::string, boost::filesystem::path> registeredGrammar;

    void handleXIncludeAndCDATA(xercesc::DOMElement *&e, std::vector<boost::filesystem::path> *dependencies=nullptr);
};

}

namespace {

template<class T>
void CheckSize<T>::check(const xercesc::DOMElement *me, const T &value, int r, int c) {}
template<class T>
void CheckSize<std::vector<T>>::check(const xercesc::DOMElement *me, const std::vector<T> &value, int r, int c) {
  if(r!=0 && r!=static_cast<int>(value.size()))
    throw MBXMLUtils::DOMEvalException("Expected vector of size "+fmatvec::toString(r)+
                           " but got vector of size "+fmatvec::toString(value.size())+".", me);
}
template<class T>
void CheckSize<std::vector<std::vector<T>>>::check(const xercesc::DOMElement *me, const std::vector<std::vector<T>> &value, int r, int c) {
  if(r!=0 && r!=static_cast<int>(value.size()))
    throw MBXMLUtils::DOMEvalException("Expected matrix of row-size "+fmatvec::toString(r)+
                           " but got matrix of row-size "+fmatvec::toString(value.size())+".", me);
  if(!value.empty() && c!=0 && c!=static_cast<int>(value[0].size()))
    throw MBXMLUtils::DOMEvalException("Expected matrix of col-size "+fmatvec::toString(c)+
                           " but got matrix of col-size "+fmatvec::toString(value[0].size())+".", me);
}

}

#endif
