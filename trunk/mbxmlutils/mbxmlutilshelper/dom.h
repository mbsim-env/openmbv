#ifndef _MBXMLUTILSHELPER_DOM_H_
#define _MBXMLUTILSHELPER_DOM_H_

#include <fmatvec/atom.h>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <sstream>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <boost/filesystem.hpp>
#include <boost/type_traits/conditional.hpp>
#include <boost/type_traits/is_same.hpp>
#include <xercesc/util/XMLString.hpp>
#include <xercesc/dom/DOMErrorHandler.hpp>
#include <xercesc/dom/DOMElement.hpp>
#include <xercesc/dom/DOMText.hpp>
#include <xercesc/dom/DOMLSParser.hpp>
#include <xercesc/dom/DOMLocator.hpp>
#include <xercesc/dom/DOMUserDataHandler.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/TransService.hpp>
#include <xercesc/framework/psvi/PSVIHandler.hpp>

namespace XERCES_CPP_NAMESPACE {
  class DOMProcessingInstruction;
  class DOMImplementation;
  class AbstractDOMParser;
}

namespace MBXMLUtils {

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
      store.push_back(boost::shared_ptr<const XMLCh>(unicode, &releaseXMLCh));
      return unicode;
    }
    const XMLCh *operator()(const std::string &str) { return operator%(str); }
    std::string operator%(const XMLCh *unicode) {
      if(!unicode || unicode[0]==0)
        return std::string();
      return reinterpret_cast<const char*>(xercesc::TranscodeToStr(unicode, "UTF8").str());
    }
    std::string operator()(const XMLCh *unicode) { return operator%(unicode); }
  private:
    static void releaseXMLCh(const XMLCh *s) { xercesc::XMLPlatformUtils::fgMemoryManager->deallocate(const_cast<XMLCh*>(s)); }
    std::vector<boost::shared_ptr<const XMLCh> > store;
};

//! Print DOM error messages
class DOMErrorPrinter: public xercesc::DOMErrorHandler, virtual public fmatvec::Atom
{
  public:
    DOMErrorPrinter() : warningCount(0), errorCount(0) {}
    bool handleError(const xercesc::DOMError&);
    int getNumWarnings() const { return warningCount; }
    int getNumErrors() const { return errorCount; }
    void resetCounter() { warningCount=0; errorCount=0; }
  protected:
    int warningCount;
    int errorCount;
};

//! Full qualified name.
//! A std::pair representing a full qualified XML name as a pair of namespace URI and local name.
class FQN : public std::pair<std::string, std::string> {
  public:
    //! Empty FQN
    FQN() : std::pair<std::string, std::string>() {}
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
    NamespaceURI(const std::string &nsuri_) : nsuri(nsuri_) {}
    FQN operator%(const std::string &localName) const { return FQN(nsuri, localName); }
    std::string getNamespaceURI() const { return nsuri; }
  private:
    std::string nsuri;
};

//! Declaration of the XML xinclude prefix/URI.
const NamespaceURI XINCLUDE("http://www.w3.org/2001/XInclude");
//! Declaration of the MBXMLUtils physicalvariable namespace prefix/URI.
const NamespaceURI PV("http://openmbv.berlios.de/MBXMLUtils/physicalvariable");

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
    //! Get first child processing instruction of the specified target
    const xercesc::DOMProcessingInstruction *getFirstProcessingInstructionChildNamed(const std::string &target) const;
    //! Get first child processing instruction of the specified target
    xercesc::DOMProcessingInstruction *getFirstProcessingInstructionChildNamed(const std::string &target);
    //! Get first child text
    const xercesc::DOMText *getFirstTextChild() const;
    //! Get first child text
    xercesc::DOMText *getFirstTextChild();
    //! Check if the element is of type base
    //! Note DOMTypeInfo::isDerivedFrom is not implemented in xerces-c hence we define our one methode here.
    bool isDerivedFrom(const FQN &base) const;
    //! Get the base URI.
    //! Returns the value of the first xml:base attribute define by a parent element or the document uri if no such element is found.
    //! If skipThis is false the xml:base attribute of this element in honored else
    //! this element is skipped and only parent elements are honored.
    //! If a xml:base attribute was found 'found' is set to this element else 'found' is NULL.
    boost::filesystem::path getOriginalFilename(bool skipThis=false,
                                                const xercesc::DOMElement *&found=DOMElementWrapper<DOMElementType>::dummyArg) const;
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
    //! Get the line number of the original element
    int getOriginalElementLineNumber() const;
    //! Set the line number of the original element
    void setOriginalElementLineNumber(int lineNr);
    //! Get attribute named name.
    std::string getAttribute(const FQN &name) const;
    //! Get attribute node named name.
    const xercesc::DOMAttr* getAttributeNode(const FQN &name) const;
    //! Get attribute node named name.
    xercesc::DOMAttr* getAttributeNode(const FQN &name);
    //! Set attribute.
    void setAttribute(const FQN &name, const std::string &value);
    //! check if this element has a attibute named name.
    bool hasAttribute(const FQN &name) const;
    //! Workaround: convert default attributes to normal attributes (must be used before importNode to also import default attributes)
    void workaroundDefaultAttributesOnImportNode();
    //! Treat this object as a pointer (like DOMElement*)
    typename boost::conditional<boost::is_same<DOMElementType, const xercesc::DOMElement>::value,
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
DOMElementWrapper<DOMElementType> E(boost::shared_ptr<DOMElementType> me) { return DOMElementWrapper<DOMElementType>(me.get()); }

//! Helper class for extending DOMAttr (use the function A(...)).
template<typename DOMAttrType>
class DOMAttrWrapper {
  public:
    //! Wrap DOMAttr to my special element
    DOMAttrWrapper(DOMAttrType *me_) : me(me_) {}
    //! Check if the element is of type base
    //! Note DOMTypeInfo::isDerivedFrom is not implemented in xerces-c hence we define our one methode here.
    bool isDerivedFrom(const FQN &base) const;
    //! Treat this object as a pointer (like DOMAttr*)
    typename boost::conditional<boost::is_same<DOMAttrType, const xercesc::DOMAttr>::value,
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
DOMAttrWrapper<DOMAttrType> A(boost::shared_ptr<DOMAttrType> me) { return DOMAttrWrapper<DOMAttrType>(me.get()); }

//! Helper class for extending DOMDocument (use the function D(...)).
template<typename DOMDocumentType>
class DOMDocumentWrapper {
  public:
    //! Wrap DOMDocument to my special element
    DOMDocumentWrapper(DOMDocumentType *me_) : me(me_) {}
    //! (re)validate the document using the parser this document was created
    void validate();
    //! create element with the given FQN
    xercesc::DOMElement* createElement(const FQN &name);
    //! Treat this object as a pointer (like DOMDocument*)
    typename boost::conditional<boost::is_same<DOMDocumentType, const xercesc::DOMDocument>::value,
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
DOMDocumentWrapper<DOMDocumentType> D(boost::shared_ptr<DOMDocumentType> me) { return DOMDocumentWrapper<DOMDocumentType>(me.get()); }

//! Helper class for DOMEvalException.
//! Extension of DOMLocator with a embed count
class EmbedDOMLocator : public xercesc::DOMLocator {
  public:
    EmbedDOMLocator(const boost::filesystem::path &file_, int row_, int embedCount_=0);
    EmbedDOMLocator(const EmbedDOMLocator &src) {
      file=x%(X()%src.file); row=src.row; embedCount=src.embedCount;
    }
    EmbedDOMLocator& operator=(const EmbedDOMLocator &src) {
      file=x%(X()%src.file); row=src.row; embedCount=src.embedCount;
      return *this;
    }
    XMLFileLoc getLineNumber() const { return row; }
    XMLFileLoc getColumnNumber() const { return 0; }
    XMLFilePos getByteOffset() const { return ~(XMLFilePos(0)); }
    XMLFilePos getUtf16Offset() const { return ~(XMLFilePos(0)); }
    xercesc::DOMNode *getRelatedNode() const { return NULL; }
    const XMLCh *getURI() const { return file; }
    std::string getEmbedCount() const;
  private:
    X x;
    const XMLCh *file;
    int row;
    int embedCount;
};

// Exception wrapping for DOMEvalException.
// Catch a exception of type DOMEvalException, set the current XML context to DOMElement e and rethrow.
#define MBXMLUTILS_RETHROW(e) \
  catch(MBXMLUtils::DOMEvalException &ex) { \
    ex.setContext(e); \
    throw; \
  }

//! Exception during evaluation of the DOM tree including a location stack
class DOMEvalException : public std::exception {
  public:
    DOMEvalException(const std::string &errorMsg_, const xercesc::DOMElement *e=NULL);
    DOMEvalException(const DOMEvalException &src) : errorMsg(src.errorMsg),
      locationStack(src.locationStack) {}
    DOMEvalException &operator=(const DOMEvalException &src) {
      errorMsg=src.errorMsg; locationStack=src.locationStack;
      return *this;
    }
    ~DOMEvalException() throw() {}
    void setContext(const xercesc::DOMElement *e);
    const std::string& getMessage() const { return errorMsg; }
    const std::vector<EmbedDOMLocator>& getLocationStack() const { return locationStack; }
    const char* what() const throw();
  private:
    std::string errorMsg;
    std::vector<EmbedDOMLocator> locationStack;
    mutable std::string whatStr;
};

//! A list of DOMEvalException
class DOMEvalExceptionList : public std::vector<DOMEvalException>, public std::exception {
  public:
    DOMEvalExceptionList() {}
    DOMEvalExceptionList(const DOMEvalExceptionList &src) : std::vector<DOMEvalException>(src), std::exception(src) {}
    DOMEvalExceptionList &operator=(const DOMEvalExceptionList &src) {
      operator=(src);
      return *this;
    }
    ~DOMEvalExceptionList() throw() {}
    const char* what() const throw();
  private:
    mutable std::string whatStr;
};

class DOMParser;

class LocationInfoFilter : public xercesc::DOMLSParserFilter {
  public:
    void setParser(DOMParser *parser_) { parser=parser_; }
    xercesc::DOMLSParserFilter::FilterAction acceptNode(xercesc::DOMNode *n);
    xercesc::DOMLSParserFilter::FilterAction startElement(xercesc::DOMElement *e);
    xercesc::DOMNodeFilter::ShowType getWhatToShow() const;
  private:
    DOMParser *parser;
    static const std::string lineNumberKey;
};

class TypeDerivativeHandler : public xercesc::PSVIHandler {
  public:
    void setParser(DOMParser *parser_) { parser=parser_; }
    void handleElementPSVI(const XMLCh *const localName, const XMLCh *const uri, xercesc::PSVIElement *elementInfo);
    void handleAttributesPSVI(const XMLCh *const localName, const XMLCh *const uri, xercesc::PSVIAttributeList *psviAttributes);
  private:
    DOMParser *parser;
};

class DOMParserUserDataHandler : public xercesc::DOMUserDataHandler {
  public:
    void handle(DOMOperationType operation, const XMLCh* const key, void *data, const xercesc::DOMNode *src, xercesc::DOMNode *dst);
};

//! A XML DOM parser.
class DOMParser {
  friend bool isDerivedFrom(const xercesc::DOMNode *me, const FQN &baseTypeName);
  friend class TypeDerivativeHandler;
  friend class LocationInfoFilter;
  friend class DOMParserUserDataHandler;
  template<typename> friend class DOMDocumentWrapper;
  public:
    //! Create DOM parser
    static boost::shared_ptr<DOMParser> create(bool validate);
    //! Load XML Schema grammar file
    void loadGrammar(const boost::filesystem::path &schemaFilename);
    //! Parse a XML document
    boost::shared_ptr<xercesc::DOMDocument> parse(const boost::filesystem::path &inputSource);
    //! Serialize a document.
    //! Helper function to write a node. This normalized the document before.
    static void serialize(xercesc::DOMNode *n, const boost::filesystem::path &outputSource, bool prettyPrint=true);
    //! reset all loaded grammars
    void resetCachedGrammarPool();
    //! create a empty document
    boost::shared_ptr<xercesc::DOMDocument> createDocument();
  private:
    static const std::string domParserKey;

    xercesc::DOMImplementation *domImpl;
    DOMParser(bool validate_);
    boost::shared_ptr<xercesc::DOMLSParser> parser;
    bool validate;
    boost::weak_ptr<DOMParser> me;
    std::map<FQN, xercesc::XSTypeDefinition*> typeMap;
    DOMErrorPrinter errorHandler;
    LocationInfoFilter locationFilter;
    TypeDerivativeHandler typeDerHandler;
    static DOMParserUserDataHandler userDataHandler;

    void handleXIncludeAndCDATA(xercesc::DOMElement *&e);
};

}

#endif
