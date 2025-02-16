#ifndef _MBXMLUTILSHELPER_DOM_H_
#define _MBXMLUTILSHELPER_DOM_H_

#include <fmatvec/atom.h>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <set>
#include <variant>
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
#include <boost/container/small_vector.hpp>
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

template<class T> struct XercesUniquePtrDeleter { void operator()(T *n) { if(n) n->release(); } };
//! A std::unique_ptr for xerces objects which calls release() to deallocate the object
template<class T> using XercesUniquePtr = std::unique_ptr<T, XercesUniquePtrDeleter<T>>;

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
      store.emplace_back(unicode, &releaseXMLCh);
      return unicode;
    }
    const XMLCh *operator()(const std::string &str) { return operator%(str); }
    std::string operator%(const XMLCh *unicode) {
      if(!unicode || unicode[0]==0)
        return {};
      return reinterpret_cast<const char*>(xercesc::TranscodeToStr(unicode, "UTF8").str());
    }
    std::string operator()(const XMLCh *unicode) { return operator%(unicode); }
    static void releaseXMLCh(const XMLCh *s) { xercesc::XMLPlatformUtils::fgMemoryManager->deallocate(const_cast<XMLCh*>(s)); }
  private:
    boost::container::small_vector<std::unique_ptr<const XMLCh, decltype(&releaseXMLCh)>, 1> store;
};

//! Full qualified name.
//! A std::pair representing a full qualified XML name as a pair of namespace URI and local name.
class FQN : public std::pair<std::string, std::string> {
  public:
    //! Empty FQN
    FQN()  = default;
    //! Anonymous FQN
    FQN(const std::string &name) : std::pair<std::string, std::string>("", name) {}
    //! Anonymous FQN (required for implicit casting of string literals to anonymous FQNs)
    FQN(const char *name) : std::pair<std::string, std::string>("", name) {}
    //! FQN
    FQN(const std::string &ns, const std::string &name) : std::pair<std::string, std::string>(ns, name) {}
};

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
    ~EmbedDOMLocator() override = default;
    EmbedDOMLocator(const EmbedDOMLocator &&src) = delete;
    EmbedDOMLocator& operator=(const EmbedDOMLocator &&src) = delete;
    XMLFileLoc getLineNumber() const override { return row; }
    XMLFileLoc getColumnNumber() const override { return 0; }
    XMLFilePos getByteOffset() const override { return ~(XMLFilePos(0)); }
    XMLFilePos getUtf16Offset() const override { return ~(XMLFilePos(0)); }
    xercesc::DOMNode *getRelatedNode() const override { return nullptr; }
    const XMLCh *getURI() const override { return file; }
    int getEmbedCount() const { return embedCount; }
    //! get a (simple) XPath expression to the location: each element is prefixed with the namespace URI between { }
    const std::string& getRootXPathExpression() const { return xpath; }
    //! get human readable (simple) XPath expression to the location
    std::string getRootHRXPathExpression() const;
    static void addNSURIPrefix(std::string nsuri, const std::vector<std::string> &prefix);
    static const std::map<std::string, std::string>& getNSURIPrefix() { return nsURIPrefix(); }
  private:
    X x;
    const XMLCh *file;
    int row;
    int embedCount;
    std::string xpath;
    static std::map<std::string, std::string>& nsURIPrefix();
};

//! Helper class to easily construct full qualified XML names (FQN) using XML namespace prefixes.
class NamespaceURI {
  public:
    NamespaceURI(std::string nsuri_, const std::vector<std::string> &preferredPrefix={}) : nsuri(std::move(nsuri_)) {
      EmbedDOMLocator::addNSURIPrefix(nsuri, preferredPrefix);
    }
    FQN operator%(const std::string &localName) const { return {nsuri, localName}; }
    const std::string& getNamespaceURI() const { return nsuri; }
  private:
    std::string nsuri;
};

//! Declaration of the XML xinclude prefix/URI.
const NamespaceURI XINCLUDE("http://www.w3.org/2001/XInclude", {"xi", "xinc", "xinclude"});
//! Declaration of the xmlns prefix/URI.
const NamespaceURI XMLNS("http://www.w3.org/2000/xmlns/", {"xmlns"});
//! Declaration of the MBXMLUtils physicalvariable namespace prefix/URI.
const NamespaceURI PV("http://www.mbsim-env.de/MBXMLUtils", {"p", "pv", "mbxmlutils"});
//! Declaration of the XML catalog namespace
const NamespaceURI XMLCATALOG("urn:oasis:names:tc:entity:xmlns:xml:catalog", {"catalog", "xmlcatalog"});

// Exception wrapping for DOMEvalException.
// Rethrow a exception as DOMEvalException with context e, a DOMEvalException is just rethrown unchanged.
#define RETHROW_AS_DOMEVALEXCEPTION(e) \
  catch(MBXMLUtils::DOMEvalException &ex) { \
    throw ex; \
  } \
  catch(const std::exception &ex) { \
    throw DOMEvalException(ex.what(), e); \
  }

//! Exception during evaluation of the DOM tree including a location stack.
//! The location stack is generated by the context node n passed to the ctor.
class DOMEvalException : public std::exception {
  friend class DOMParser;
  friend class DOMErrorPrinter;
  public:
    DOMEvalException() = default;
    DOMEvalException(const std::string &errorMsg_, const xercesc::DOMNode *n);
    void appendContext(const xercesc::DOMNode *n, int lineNr=0);
    const std::string& getMessage() const { return errorMsg; }
    void setMessage(const std::string& errorMsg_) { errorMsg=errorMsg_; }
    void setSubsequentError(bool sse) { subsequentError=sse; }
    const char* what() const noexcept override;
    xercesc::DOMNode::NodeType getNodeType() const { return nodeType; }
    static bool isHTMLOutputEnabled();
    static void htmlEscaping(std::string &msg);
  protected:
    DOMEvalException(const std::string &errorMsg_, const xercesc::DOMLocator &loc);
  private:
    /** Convert a error location and error message for outputting it to the console.
      The behaviour of this function can be adapted by the environment variable MBXMLUTILS_ERROROUTPUT.
      For each error the value of this variable is printed, where the value is interpreted as a
      "Boost-Extended Format String Syntax", where the following named sub-expressions are recognized:
     
      named sub-expr  | value of the named sub-expression
      --------------- | -------------------------------------------
      msg             | the error message
      file            | the filename where the error occurred (may be relative to the current directory)
      absfile         | same as file but always absolute
      urifile         | same as absfile but URI encoded
      line            | the line number in the file where the error occurred
      xpath           | the XPath expression from the root element of the file to the element in the file where the error occurred
      ecount          | the embed count number where the error occurred
      sse             | undefined value but only defined if this is a subsequent error
     
      All these named sub-expressions may not be defined (see "Boost-Extended Format Syntax Syntax" on how to handle this).
      If the environment variable MBXMLUTILS_ERROROUTPUT is not set then the default GCC is used.
     
      The following values for MBXMLUTILS_ERROROUTPUT are interpreted as an internally defined expression:
      GCC: use gcc style output with color and link escape sequences when stdout is a tty
      GCCTTY use gcc style output always with color and link escape sequences
      GCCNONE use gcc style output without any escape sequences
      HTMLFILELINE: use HTML like output with a link with filename and line number
      HTMLXPATH: use HTML like output with a link with filename and xpath expression
     */
    static std::string convertToString(const EmbedDOMLocator &loc, const std::string &message, bool subsequentError=false);

    bool subsequentError{false};
    std::string errorMsg;
    std::vector<EmbedDOMLocator> locationStack;
    mutable std::string whatStr;
    xercesc::DOMNode::NodeType nodeType { static_cast<xercesc::DOMNode::NodeType>(-1) };
};

//! Print DOM error messages
class DOMErrorPrinter: public xercesc::DOMErrorHandler, virtual public fmatvec::Atom
{
  public:
    DOMErrorPrinter() = default;
    bool handleError(const xercesc::DOMError&) override;
    bool hasError() { return errorSet; }
    const DOMEvalException& getError() { return error; }
    void resetError() { errorSet=false; }
  private:
    bool errorSet{false};
    DOMEvalException error;
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
    //! Add a processing instruction child of the specified target
    void addProcessingInstructionChildNamed(const std::string &target, const std::string &data);
    //! Get the embed data named name from the current element. Returns "" if not such data exists.
    std::string getEmbedData(const std::string &name) const;
    //! Add a embed data named name to the current element holding data.
    //! If the element has already such a name embedData it is overwritten with data
    void addEmbedData(const std::string &name, const std::string &data);
    //! Get first child comment
    const xercesc::DOMComment *getFirstCommentChild() const;
    //! Get first child comment
    xercesc::DOMComment *getFirstCommentChild();
    //! Get first none empty child text or CDATA node.
    //! A empty text node (only spaces, tabs and new-lines) is interpreted as a formatting node not a content node.
    //! If, however, only empty text or CDATA nodes exist than the last empty text or CDATA node is returned.
    //! If no text or CDATA node or more then one none empty text or CDATA node exists nullptr is returned.
    const xercesc::DOMText *getFirstTextChild() const;
    //! Get first none empty child text or CDATA node.
    //! A empty text node (only spaces, tabs and new-lines) is interpreted as a formatting node not a content node.
    //! If, however, only empty text or CDATA nodes exist than the last empty text or CDATA node is returned.
    //! If no text or CDATA node or more then one none empty text or CDATA node exists nullptr is returned.
    xercesc::DOMText *getFirstTextChild();
    //! Get the child text as type T
    template<class T> T getText(int r=0, int c=0) const {
      try {
        auto textEle=E(me)->getFirstTextChild();
        auto text=textEle ? X()%textEle->getData() : "";
        auto ret=boost::lexical_cast<T>(text);
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
    //! If orgFileName is given then its different: its just the given filename set as OriginalFileName.
    void setOriginalFilename(boost::filesystem::path orgFileName=boost::filesystem::path());
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
    //! The root element may not be the document itself if embedding has occurred.
    //! Embedding is preserved by this XPath.
    //! The output is /<root-element-name>[1]/<element-name>[<element-nr>]/<element-name>[<element-nr>] ...
    std::string getRootXPathExpression() const;
    //! Get a (fast) unique location identified of this element relative to the DOMDocument.
    //! Use DOMDocumentWrapper::locateElement(...) to get the same element in another document.
    std::vector<int> getElementLocation() const;
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
DOMElementWrapper<DOMElementType> E(const std::shared_ptr<DOMElementType> &me) { return DOMElementWrapper<DOMElementType>(me.get()); }
//! Helper function, with a very short name, for automatic type deduction for DOMElementWrapper.
template<typename DOMElementType>
DOMElementWrapper<DOMElementType> E(const XercesUniquePtr<DOMElementType> &me) { return DOMElementWrapper<DOMElementType>(me.get()); }

template<> const xercesc::DOMElement *DOMElementWrapper<      xercesc::DOMElement>::dummyArg;
template<> const xercesc::DOMElement *DOMElementWrapper<const xercesc::DOMElement>::dummyArg;

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
    //! The root element may not be the document itself if embedding has occurred.
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
DOMAttrWrapper<DOMAttrType> A(const std::shared_ptr<DOMAttrType> &me) { return DOMAttrWrapper<DOMAttrType>(me.get()); }
//! Helper function, with a very short name, for automatic type deduction for DOMAttrWrapper.
template<typename DOMAttrType>
DOMAttrWrapper<DOMAttrType> A(const XercesUniquePtr<DOMAttrType> &me) { return DOMAttrWrapper<DOMAttrType>(me.get()); }

class DOMParser;

//! Helper class for extending DOMDocument (use the function D(...)).
template<typename DOMDocumentType>
class DOMDocumentWrapper {
  public:
    //! Wrap DOMDocument to my special element
    DOMDocumentWrapper(DOMDocumentType *me_) : me(me_) {}
    //! (re)validate the document using the parser this document was created
    //! This will renew ALL nodes in the document.
    //! But the old root DOMElement, including all old childrens, will be returned. If you don't use
    //! the return value it will be released automatically.
    XercesUniquePtr<xercesc::DOMElement> validate();
    //! create element with the given FQN
    //! Note: a empty namespace (name.first.empty()==true) as no namespace
    xercesc::DOMElement* createElement(const FQN &name);
    //! Get full qualified tag name
    std::shared_ptr<DOMParser> getParser() const;
    //! Get the filename of the document.
    //! This is the same as getDocumentURI but with the file schema removed and is a relative path if it was read by a relative path
    boost::filesystem::path getDocumentFilename() const;
    //! Get the node (DOMElement or DOMAttrType) corresponding the given xpathExpression relative to the root.
    //! If context is nullptr than the root element is used.
    //! Only a very small subset of XPath is supported by this function:
    //! the output of getRootXPathExpression
    xercesc::DOMNode* evalRootXPathExpression(std::string xpathExpression, xercesc::DOMElement* context=nullptr);
    //! Get the element in this document which corresponds to the location idx.
    //! idx is usualy retrieved via DOMElementWrapper::getElementLocation().
    xercesc::DOMElement* locateElement(const std::vector<int> &idx) const;
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
DOMDocumentWrapper<DOMDocumentType> D(const std::shared_ptr<DOMDocumentType> &me) { return DOMDocumentWrapper<DOMDocumentType>(me.get()); }
//! Helper function, with a very short name, for automatic type deduction for DOMDocumentWrapper.
template<typename DOMDocumentType>
DOMDocumentWrapper<DOMDocumentType> D(const XercesUniquePtr<DOMDocumentType> &me) { return DOMDocumentWrapper<DOMDocumentType>(me.get()); }

class LocationInfoFilter : public xercesc::DOMLSParserFilter {
  public:
    void setParser(DOMParser *parser_) { parser=parser_; }
    xercesc::DOMLSParserFilter::FilterAction acceptNode(xercesc::DOMNode *n) override;
    xercesc::DOMLSParserFilter::FilterAction startElement(xercesc::DOMElement *e) override;
    xercesc::DOMNodeFilter::ShowType getWhatToShow() const override;
    void setLineNumberOffset(int offset) { lineNumberOffset=offset; }
  private:
    DOMParser *parser;
    int lineNumberOffset { 0 };
};

class TypeDerivativeHandler : public xercesc::PSVIHandler, virtual public fmatvec::Atom {
  public:
    void setParser(DOMParser *parser_) { parser=parser_; }
    void handleElementPSVI(const XMLCh *localName, const XMLCh *uri, xercesc::PSVIElement *info) override;
    void handleAttributesPSVI(const XMLCh *localName, const XMLCh *uri, xercesc::PSVIAttributeList *psviAttributes) override;
  private:
    DOMParser *parser;
};

class UserDataHandler : public xercesc::DOMUserDataHandler {
  public:
    void handle(DOMOperationType operation, const XMLCh* key, void *data, const xercesc::DOMNode *src, xercesc::DOMNode *dst) override;
};

extern UserDataHandler userDataHandler;

class EntityResolver : public xercesc::XMLEntityResolver {
  public:
    void setParser(DOMParser *parser_) { parser=parser_; }
    xercesc::InputSource* resolveEntity(xercesc::XMLResourceIdentifier *resourceIdentifier) override;
  private:
    DOMParser *parser;
};

//! A XML DOM parser.
class DOMParser : public std::enable_shared_from_this<DOMParser> {
  friend class TypeDerivativeHandler;
  friend class LocationInfoFilter;
  friend class UserDataHandler;
  friend class EntityResolver;
  friend class DOMElementWrapper<xercesc::DOMElement>;
  friend class DOMElementWrapper<const xercesc::DOMElement>;
  template<typename> friend class DOMDocumentWrapper;
  public:
    //! Create DOM parser
    //! A none validating parser if xmlCatalog is empty or a nullptr.
    //! A validating parser if xmlCatalog defines a XML catalog file or a root XML element of a catalog
    static std::shared_ptr<DOMParser> create(const std::variant<boost::filesystem::path, xercesc::DOMElement*> &xmlCatalog=static_cast<xercesc::DOMElement*>(nullptr));
    //! Parse a XML document from a filename.
    //! Track file dependencies if dependencies is not null.
    //! Allow XML XInclude if doXInclude is true.
    std::shared_ptr<xercesc::DOMDocument> parse(const boost::filesystem::path &inputSource,
                                                std::vector<boost::filesystem::path> *dependencies=nullptr,
                                                bool doXInclude=true);
    //! Parse a XML document from a input stream.
    //! Track file dependencies if dependencies is not null.
    //! Allow XML XInclude if doXInclude is true.
    std::shared_ptr<xercesc::DOMDocument> parse( std::istream &inputStream,
                                                std::vector<boost::filesystem::path> *dependencies=nullptr,
                                                bool doXInclude=true);
    //! Parse a XML document from a istream to a given context.
    //! Track file dependencies if dependencies is not null.
    //! Allow XML XInclude if doXInclude is true.
    xercesc::DOMElement* parseWithContext(const std::string &str, xercesc::DOMNode *contextNode, xercesc::DOMLSParser::ActionType action,
                                          std::vector<boost::filesystem::path> *dependencies=nullptr,
                                          bool doXInclude=true);
    //! Serialize a document to a file.
    //! Helper function to write a node.
    static void serialize(xercesc::DOMNode *n, const boost::filesystem::path &outputSource);
    //! Serialize a document to a memory (std::string).
    //! Helper function to write a node.
    static void serializeToString(xercesc::DOMNode *n, std::string &outputData);
    //! reset all loaded grammars
    void resetCachedGrammarPool();
    //! create a empty document
    std::shared_ptr<xercesc::DOMDocument> createDocument();
    
    const std::map<FQN, xercesc::XSTypeDefinition*>& getTypeMap() const { return typeMap; }
  private:
    xercesc::DOMImplementation *domImpl;
    DOMParser(const std::variant<boost::filesystem::path, xercesc::DOMElement*> &xmlCatalog);
    std::shared_ptr<xercesc::DOMLSParser> parser;
    std::map<FQN, xercesc::XSTypeDefinition*> typeMap;
    DOMErrorPrinter errorHandler;
    LocationInfoFilter locationFilter;
    TypeDerivativeHandler typeDerHandler;
    EntityResolver entityResolver;
    std::map<std::string, boost::filesystem::path> registeredGrammar;

    static void handleXInclude(xercesc::DOMElement *&e, std::vector<boost::filesystem::path> *dependencies);
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
