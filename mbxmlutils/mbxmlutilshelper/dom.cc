#include <config.h>
#include "dom.h"
#include <boost/bind.hpp>
#include <boost/make_shared.hpp>
#include <boost/format.hpp>
#include <boost/locale.hpp>
#include <boost/lexical_cast.hpp>
#include <xercesc/dom/DOMImplementationRegistry.hpp>
#include <xercesc/dom/DOMImplementation.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMLSSerializer.hpp>
#include <xercesc/dom/DOMAttr.hpp>
#include <xercesc/dom/DOMTypeInfo.hpp>
#include <xercesc/dom/DOMProcessingInstruction.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/framework/psvi/PSVIElement.hpp>
#include <xercesc/framework/psvi/PSVIAttributeList.hpp>
#include <xercesc/framework/MemBufInputSource.hpp>
#include <xercesc/framework/Wrapper4InputSource.hpp>
#include "getinstallpath.h"

// we need some internal xerces classes (here the XMLScanner to get the current line number during parsing)
#include <xercesc/internal/XMLScanner.hpp>

using namespace std;
using namespace xercesc;
using namespace boost;
using namespace boost::filesystem;

namespace MBXMLUtils {

namespace {
  InitXerces initXerces;

  //TODO not working on Windows
  //TODO // NOTE: we can skip the use of utf8Facet (see below) and set the facet globally (for bfs::path and others) using:
  //TODO // std::locale::global(boost::locale::generator().generate("UTF8"));
  //TODO // boost::filesystem::path::imbue(std::locale());
  //TODO const path::codecvt_type *utf8Facet(&use_facet<path::codecvt_type>(boost::locale::generator().generate("UTF8")));
  #define CODECVT

  // START: ugly hack to call a protected/private method from outside
  // (from http://bloglitb.blogspot.de/2010/07/access-to-private-members-thats-easy.html)
  template<typename Tag>
  struct result {
    typedef typename Tag::type type;
    static type ptr;
  };
  
  template<typename Tag>
  typename result<Tag>::type result<Tag>::ptr;
  
  template<typename Tag, typename Tag::type p>
  struct rob : result<Tag> {
    struct filler {
      filler() { result<Tag>::ptr = p; }
    };
    static filler filler_obj;
  };
  
  template<typename Tag, typename Tag::type p>
  typename rob<Tag, p>::filler rob<Tag, p>::filler_obj;
  // END: ugly hack to call a protected/private method from outside
  // (from http://bloglitb.blogspot.de/2010/07/access-to-private-members-thats-easy.html)

  path toRelativePath(path absPath, path relTo=current_path()) {
    if(!absPath.is_absolute())
      throw runtime_error("First argument of toRelativePath must be a absolute path.");
    path::iterator curIt, absIt;
    for(curIt=relTo.begin(), absIt=absPath.begin(); curIt!=relTo.end() && *curIt==*absIt; ++curIt, ++absIt);
    if(curIt==relTo.end()) {
      path relPathRet;
      for(; absIt!=absPath.end(); ++absIt)
        relPathRet/=*absIt;
      return relPathRet;
    }
    return absPath;
  }
}

bool DOMErrorPrinter::handleError(const DOMError& e)
{
  string type;
  switch(e.getSeverity()) {
    case DOMError::DOM_SEVERITY_WARNING:     type="Warning";     warningCount++; break;
    case DOMError::DOM_SEVERITY_ERROR:       type="Error";       errorCount++;   break;
    case DOMError::DOM_SEVERITY_FATAL_ERROR: type="Fatal error"; errorCount++;   break;
  }
  DOMLocator *loc=e.getLocation();
  // Note we print here all message types to the Warn stream. If a error occurred a exception is thrown later
  msg(Warn)<<DOMEvalException::fileOutput(*loc)<<": "<<type<<": "<<X()%e.getMessage()<<endl;
  return e.getSeverity()!=DOMError::DOM_SEVERITY_FATAL_ERROR; // continue parsing for none fatal errors
}

EmbedDOMLocator::EmbedDOMLocator(const path &file_, int row_, int embedCount_) : row(row_), embedCount(embedCount_) {
  file=x%file_.string(CODECVT);
}

std::string EmbedDOMLocator::getEmbedCount() const {
  if(embedCount>0) {
    stringstream str;
    str<<"[count="<<embedCount<<"]";
    return str.str();
  }
  return "";
}

template<> const DOMElement *DOMElementWrapper<      DOMElement>::dummyArg=NULL;
template<> const DOMElement *DOMElementWrapper<const DOMElement>::dummyArg=NULL;

template<typename DOMElementType>
const DOMElement *DOMElementWrapper<DOMElementType>::getFirstElementChildNamed(const FQN &name) const {
  for(DOMElement *ret=me->getFirstElementChild(); ret; ret=ret->getNextElementSibling())
    if(E(ret)->getTagName()==name)
      return ret;
  return NULL;
};
template const DOMElement *DOMElementWrapper<const DOMElement>::getFirstElementChildNamed(const FQN &name) const; // explicit instantiate const variant

template<typename DOMElementType>
DOMElement *DOMElementWrapper<DOMElementType>::getFirstElementChildNamed(const FQN &name) {
  for(DOMElement *ret=me->getFirstElementChild(); ret; ret=ret->getNextElementSibling())
    if(E(ret)->getTagName()==name)
      return ret;
  return NULL;
};

template<typename DOMElementType>
const DOMProcessingInstruction *DOMElementWrapper<DOMElementType>::getFirstProcessingInstructionChildNamed(const string &target) const {
  for(DOMNode *ret=me->getFirstChild(); ret; ret=ret->getNextSibling()) {
    if(ret->getNodeType()!=DOMNode::PROCESSING_INSTRUCTION_NODE)
      continue;
    if(X()%static_cast<DOMProcessingInstruction*>(ret)->getTarget()==target)
      return static_cast<DOMProcessingInstruction*>(ret);
  }
  return NULL;
}
template const DOMProcessingInstruction *DOMElementWrapper<const DOMElement>::getFirstProcessingInstructionChildNamed(const string &target) const; // explicit instantiate const variant

template<typename DOMElementType>
DOMProcessingInstruction *DOMElementWrapper<DOMElementType>::getFirstProcessingInstructionChildNamed(const string &target) {
  for(DOMNode *ret=me->getFirstChild(); ret; ret=ret->getNextSibling()) {
    if(ret->getNodeType()!=DOMNode::PROCESSING_INSTRUCTION_NODE)
      continue;
    if(X()%static_cast<DOMProcessingInstruction*>(ret)->getTarget()==target)
      return static_cast<DOMProcessingInstruction*>(ret);
  }
  return NULL;
}

template<typename DOMElementType>
const DOMText *DOMElementWrapper<DOMElementType>::getFirstTextChild() const {
  for(DOMNode *ret=me->getFirstChild(); ret; ret=ret->getNextSibling())
    if(ret->getNodeType()==DOMNode::TEXT_NODE)
      return static_cast<DOMText*>(ret);
  return NULL;
}
template const DOMText *DOMElementWrapper<const DOMElement>::getFirstTextChild() const; // explicit instantiate const variant

template<typename DOMElementType>
DOMText *DOMElementWrapper<DOMElementType>::getFirstTextChild() {
  for(DOMNode *ret=me->getFirstChild(); ret; ret=ret->getNextSibling())
    if(ret->getNodeType()==DOMNode::TEXT_NODE)
      return static_cast<DOMText*>(ret);
  return NULL;
}

template<typename DOMElementType>
path DOMElementWrapper<DOMElementType>::getOriginalFilename(bool skipThis, const DOMElement *&found) const {
  found=NULL;
  const DOMElement *e;
  if(skipThis)
    e=me->getParentNode()->getNodeType()==DOMNode::ELEMENT_NODE?static_cast<DOMElement*>(me->getParentNode()):NULL;
  else
    e=me;
  while(e) {
    if(E(e)->getFirstProcessingInstructionChildNamed("OriginalFilename")) {
      if(e->getOwnerDocument()->getDocumentElement()!=e)
        found=e;
      return X()%E(e)->getFirstProcessingInstructionChildNamed("OriginalFilename")->getData();
    }
    e=e->getParentNode()->getNodeType()==DOMNode::ELEMENT_NODE?static_cast<DOMElement*>(e->getParentNode()):NULL;
  }
  string uri=X()%me->getOwnerDocument()->getDocumentURI();
  static const string fileScheme="file://";
  if(uri.substr(0, fileScheme.length())!=fileScheme)
    throw runtime_error("Only local filenames are allowed.");
  return uri.substr(fileScheme.length());
}
template path DOMElementWrapper<const DOMElement>::getOriginalFilename(bool skipThis, const DOMElement *&found) const; // explicit instantiate const variant

template<typename DOMElementType>
path DOMElementWrapper<DOMElementType>::convertPath(const path &relPath) const {
  if(relPath.is_absolute())
    return relPath;
  return toRelativePath(absolute(relPath, getOriginalFilename().parent_path()));
}
template path DOMElementWrapper<const DOMElement>::convertPath(const path &relPath) const; // explicit instantiate const variant

template<typename DOMElementType>
string DOMElementWrapper<DOMElementType>::getAttribute(const FQN &name) const {
  return X()%me->getAttributeNS(X()%name.first, X()%name.second);
}
template string DOMElementWrapper<const DOMElement>::getAttribute(const FQN &name) const; // explicit instantiate const variant

template<typename DOMElementType>
const DOMAttr* DOMElementWrapper<DOMElementType>::getAttributeNode(const FQN &name) const {
  return me->getAttributeNodeNS(X()%name.first, X()%name.second);
}
template const DOMAttr* DOMElementWrapper<const DOMElement>::getAttributeNode(const FQN &name) const; // explicit instantiate const variant

template<typename DOMElementType>
DOMAttr* DOMElementWrapper<DOMElementType>::getAttributeNode(const FQN &name) {
  return me->getAttributeNodeNS(X()%name.first, X()%name.second);
}

template<typename DOMElementType>
void DOMElementWrapper<DOMElementType>::setAttribute(const FQN &name, const string &value) {
  me->setAttributeNS(X()%name.first, X()%name.second, X()%value);
}

template<typename DOMElementType>
int DOMElementWrapper<DOMElementType>::getLineNumber() const {
  const DOMProcessingInstruction *pi=getFirstProcessingInstructionChildNamed("LineNr");
  if(pi)
    return lexical_cast<int>((X()%pi->getData()));
  return 0;
}
template int DOMElementWrapper<const DOMElement>::getLineNumber() const; // explicit instantiate const variant

template<typename DOMElementType>
int DOMElementWrapper<DOMElementType>::getEmbedCountNumber() const {
  const DOMProcessingInstruction *pi=getFirstProcessingInstructionChildNamed("EmbedCountNr");
  if(pi)
    return lexical_cast<int>((X()%pi->getData()));
  return 0;
}
template int DOMElementWrapper<const DOMElement>::getEmbedCountNumber() const; // explicit instantiate const variant

template<typename DOMElementType>
void DOMElementWrapper<DOMElementType>::setEmbedCountNumber(int embedCount) {
  stringstream str;
  str<<embedCount;
  DOMProcessingInstruction *embedCountPI=me->getOwnerDocument()->createProcessingInstruction(X()%"EmbedCountNr", X()%str.str());
  me->insertBefore(embedCountPI, me->getFirstChild());
}

template<typename DOMElementType>
int DOMElementWrapper<DOMElementType>::getOriginalElementLineNumber() const {
  const DOMProcessingInstruction *pi=getFirstProcessingInstructionChildNamed("OriginalElementLineNr");
  if(pi)
    return lexical_cast<int>((X()%pi->getData()));
  return 0;
}
template int DOMElementWrapper<const DOMElement>::getOriginalElementLineNumber() const; // explicit instantiate const variant

template<typename DOMElementType>
void DOMElementWrapper<DOMElementType>::setOriginalElementLineNumber(int lineNr) {
  stringstream str;
  str<<lineNr;
  DOMProcessingInstruction *embedCountPI=me->getOwnerDocument()->createProcessingInstruction(X()%"OriginalElementLineNr", X()%str.str());
  me->insertBefore(embedCountPI, me->getFirstChild());
}

template<typename DOMElementType>
void DOMElementWrapper<DOMElementType>::workaroundDefaultAttributesOnImportNode() {
  // rest all default attributes to the default value exlicitly: this removed the default "flag"
  DOMNamedNodeMap *attr=me->getAttributes();
  for(int i=0; i<attr->getLength(); i++) {
    DOMAttr *a=static_cast<DOMAttr*>(attr->item(i));
    if(!a->getSpecified())
      a->setValue(X()%(X()%a->getValue()));
  }
  // loop over all child elements recursively
  for(DOMElement *c=me->getFirstElementChild(); c!=0; c=c->getNextElementSibling())
    E(c)->workaroundDefaultAttributesOnImportNode();
}

template<typename DOMElementType>
bool DOMElementWrapper<DOMElementType>::hasAttribute(const FQN &name) const {
  return me->hasAttributeNS(X()%name.first, X()%name.second);
}
template bool DOMElementWrapper<const DOMElement>::hasAttribute(const FQN &name) const; // explicit instantiate const variant

bool isDerivedFrom(const DOMNode *me, const FQN &baseTypeName) {
  shared_ptr<DOMParser> parser=*static_cast<shared_ptr<DOMParser>*>(me->getOwnerDocument()->getUserData(X()%DOMParser::domParserKey));

  const DOMTypeInfo *type;
  if(me->getNodeType()==DOMNode::ELEMENT_NODE)
    type=static_cast<const DOMElement*>(me)->getSchemaTypeInfo();
  else
    type=static_cast<const DOMAttr*>(me)->getSchemaTypeInfo();
  FQN typeName(X()%type->getTypeNamespace(), X()%type->getTypeName());

  map<FQN, XSTypeDefinition*>::iterator it=parser->typeMap.find(typeName);
  if(it==parser->typeMap.end())
    throw runtime_error("Internal error: Type {"+typeName.first+"}"+typeName.second+" not found.");
  return it->second->derivedFrom(X()%baseTypeName.first, X()%baseTypeName.second);
}

template<typename DOMElementType>
bool DOMElementWrapper<DOMElementType>::isDerivedFrom(const FQN &baseTypeName) const {
  return MBXMLUtils::isDerivedFrom(me, baseTypeName);
}
template bool DOMElementWrapper<const DOMElement>::isDerivedFrom(const FQN &baseTypeName) const; // explicit instantiate const variant

// Explicit instantiate none const variante. Note the const variant should only be instantiate for const members.
template class DOMElementWrapper<DOMElement>;

template<typename DOMAttrType>
bool DOMAttrWrapper<DOMAttrType>::isDerivedFrom(const FQN &baseTypeName) const {
  return MBXMLUtils::isDerivedFrom(me, baseTypeName);
}
template bool DOMAttrWrapper<const DOMAttr>::isDerivedFrom(const FQN &baseTypeName) const; // explicit instantiate const variant

// Explicit instantiate none const variante. Note the const variant should only be instantiate for const members.
template class DOMAttrWrapper<DOMAttr>;

template<typename DOMDocumentType>
void DOMDocumentWrapper<DOMDocumentType>::validate() {
  // normalize document
  me->normalizeDocument();

  // serialize to memory
  DOMImplementation *impl=DOMImplementationRegistry::getDOMImplementation(X()%"");
  shared_ptr<DOMLSSerializer> ser(impl->createLSSerializer(), bind(&DOMLSSerializer::release, _1));
  shared_ptr<XMLCh> data(ser->writeToString(me), &X::releaseXMLCh); // serialize to data being UTF-16
  if(!data.get())
    throw runtime_error("Serializing the document to memory failed.");
  // count number of words (16bit blocks); UTF-16 multi word characters are counted as 2 words
  int dataByteLen=0;
  while(data.get()[dataByteLen]!=0) { dataByteLen++; }
  dataByteLen*=2; // a word has 2 bytes

  // parse from memory
  shared_ptr<DOMParser> parser=*static_cast<shared_ptr<DOMParser>*>(me->getUserData(X()%DOMParser::domParserKey));
  MemBufInputSource memInput(reinterpret_cast<XMLByte*>(data.get()), dataByteLen, X()%"<internal>", false);
  Wrapper4InputSource domInput(&memInput, false);
  parser->errorHandler.resetCounter();
  shared_ptr<DOMDocument> newDoc(parser->parser->parse(&domInput), bind(&DOMDocument::release, _1));
  if(parser->errorHandler.getNumErrors()>0)
    throw runtime_error(str(format("Validation failed: %1% Errors, %2% Warnings, see above.")%
      parser->errorHandler.getNumErrors()%parser->errorHandler.getNumWarnings()));

  // replace old document element with new one
  E(newDoc->getDocumentElement())->workaroundDefaultAttributesOnImportNode();// workaround
  DOMNode *newRoot=me->importNode(newDoc->getDocumentElement(), true);
  me->replaceChild(newRoot, me->getDocumentElement())->release();
}

template<typename DOMDocumentType>
xercesc::DOMElement* DOMDocumentWrapper<DOMDocumentType>::createElement(const FQN &name) {
  return me->createElementNS(X()%name.first, X()%name.second);
}

// Explicit instantiate none const variante. Note the const variant should only be instantiate for const members.
template class DOMDocumentWrapper<DOMDocument>;

DOMEvalException::DOMEvalException(const string &errorMsg_, const DOMElement *e, const DOMAttr *a) {
  // store error message
  errorMsg=errorMsg_;
  // create a DOMLocator stack (by using embed elements (OriginalFilename processing instructions))
  if(e)
    setContext(e);
  if(a)
    attrName=X()%a->getName();
}

void DOMEvalException::generateLocationStack(const xercesc::DOMElement *e, std::vector<EmbedDOMLocator> &locationStack) {
  const DOMElement *ee=e;
  const DOMElement *found;
  locationStack.clear();
  locationStack.push_back(EmbedDOMLocator(E(ee)->getOriginalFilename(false, found), E(ee)->getLineNumber(), E(ee)->getEmbedCountNumber()));
  ee=found;
  while(ee) {
    locationStack.push_back(EmbedDOMLocator(E(ee)->getOriginalFilename(true, found),
                                            E(ee)->getOriginalElementLineNumber(),
                                            E(ee)->getEmbedCountNumber()));
    ee=found;
  }
}

void DOMEvalException::locationStack2Stream(const string &indent, const vector<EmbedDOMLocator> &locationStack,
                                            const string &attrName, ostream &str) {
  if(!locationStack.empty()) {
    vector<EmbedDOMLocator>::const_iterator it=locationStack.begin();
    str<<indent<<"At "<<(attrName.empty()?"":"attribute "+attrName)<<fileOutput(*it)<<endl;
    for(it++; it!=locationStack.end(); it++)
      str<<indent<<"included by "<<fileOutput(*it)<<it->getEmbedCount()<<endl;
  }
}

string DOMEvalException::fileOutput(const DOMLocator &loc) {
  if(!getenv("MBXMLUTILS_HTMLOUTPUT"))
    // normal (ascii) output of filenames and line numbers
    return X()%loc.getURI()+":"+boost::lexical_cast<string>(loc.getLineNumber());
  else
    // html output of filenames and line numbers
    return "<a href=\""+X()%loc.getURI()+"?line="+boost::lexical_cast<string>(loc.getLineNumber())+"\">"+
      X()%loc.getURI()+":"+boost::lexical_cast<string>(loc.getLineNumber())+"</a>";
}

void DOMEvalException::setContext(const DOMElement *e) {
  generateLocationStack(e, locationStack);
}

const char* DOMEvalException::what() const throw() {
  // create return string
  stringstream str;
  str<<errorMsg<<endl;
  locationStack2Stream("", locationStack, attrName, str);
  whatStr=str.str();
  whatStr.resize(whatStr.length()-1); // remote the trailing line feed
  return whatStr.c_str();
}

const string LocationInfoFilter::lineNumberKey("http://openmbv.berlios.de/MBXMLUtils/lineNumber");

// START: call protected AbstractDOMParser::getScanner from outside, see above
struct GETSCANNER { typedef XMLScanner*(AbstractDOMParser::*type)() const; };
template class rob<GETSCANNER, &AbstractDOMParser::getScanner>;
// END: call protected AbstractDOMParser::getScanner from outside, see above
DOMLSParserFilter::FilterAction LocationInfoFilter::startElement(DOMElement *e) {
  // store the line number of the element start as user data
  AbstractDOMParser *abstractParser=dynamic_cast<AbstractDOMParser*>(parser->parser.get());
  int lineNr=(abstractParser->*result<GETSCANNER>::ptr)()->getLocator()->getLineNumber();
  e->setUserData(X()%lineNumberKey, new int(lineNr), NULL);
  return FILTER_ACCEPT;
}

DOMLSParserFilter::FilterAction LocationInfoFilter::acceptNode(DOMNode *n) {
  DOMElement* e=static_cast<DOMElement*>(n);
  // get the line number of the element start from user data and reset user data
  shared_ptr<int> lineNrPtr(static_cast<int*>(e->getUserData(X()%lineNumberKey)));
  e->setUserData(X()%lineNumberKey, NULL, NULL);
  // if no LineNr processing instruction exists create it using the line number from user data
  DOMProcessingInstruction *pi=E(e)->getFirstProcessingInstructionChildNamed("LineNr");
  if(!pi) {
    stringstream str;
    str<<*lineNrPtr;
    DOMProcessingInstruction *lineNrPI=e->getOwnerDocument()->createProcessingInstruction(X()%"LineNr", X()%str.str());
    e->insertBefore(lineNrPI, e->getFirstChild());
  }
  // return (lineNrPtr is deleted implicitly)
  return FILTER_ACCEPT;
}

DOMNodeFilter::ShowType LocationInfoFilter::getWhatToShow() const {
  return DOMNodeFilter::SHOW_ELEMENT;
}

void TypeDerivativeHandler::handleElementPSVI(const XMLCh *const localName, const XMLCh *const uri, PSVIElement *info) {
  XSTypeDefinition *type=info->getTypeDefinition();
  if(!type) {
    msg(Warn)<<"No type defined for element {"<<X()%uri<<"}"<<X()%localName<<"."<<endl;
    return;
  }
  FQN name(X()%type->getNamespace(), X()%type->getName());
  parser->typeMap[name]=type;
}

void TypeDerivativeHandler::handleAttributesPSVI(const XMLCh *const localName, const XMLCh *const uri, PSVIAttributeList *psviAttributes) {
  for(int i=0; i<psviAttributes->getLength(); i++) {
    PSVIAttribute *info=psviAttributes->getAttributePSVIAtIndex(i);
    // the xmlns attribute has not type -> skip it (maybe a bug in xerces, but this attribute is not needed)
    if(X()%psviAttributes->getAttributeNamespaceAtIndex(i)=="" && X()%psviAttributes->getAttributeNameAtIndex(i)=="xmlns")
      continue;

    XSTypeDefinition *type=info->getTypeDefinition();
    if(!type) {
      msg(Warn)<<"No type defined for attribute {"<<X()%psviAttributes->getAttributeNamespaceAtIndex(i)<<"}"
               <<X()%psviAttributes->getAttributeNameAtIndex(i)<<" in element {"<<X()%uri<<"}"<<X()%localName<<"."<<endl;
      return;
    }
    FQN name(X()%type->getNamespace(), X()%type->getName());
    parser->typeMap[name]=type;
  }
}

void DOMParserUserDataHandler::handle(DOMUserDataHandler::DOMOperationType operation, const XMLCh* const key,
  void *data, const DOMNode *src, DOMNode *dst) {
  if(X()%key!=DOMParser::domParserKey)
    throw runtime_error("Internal error: unknown user data key");
  if(operation==NODE_DELETED) {
    delete static_cast<shared_ptr<DOMParser>*>(data);
    return;
  }
  if(src->getNodeType()==DOMNode::DOCUMENT_NODE && operation==NODE_CLONED) { // src->getNodeType()!=DOCUMENT_NODE should not happen see below (maybe a xerces bug)
    // copy user data (deeply) (this incremnets the ref count on the parser)
    shared_ptr<DOMParser> parser=*static_cast<shared_ptr<DOMParser>*>(data);
    dst->setUserData(X()%DOMParser::domParserKey, new shared_ptr<DOMParser>(parser), &DOMParser::userDataHandler);
    return;
  }
  return; // we should throw here a 'throw runtime_error("Internal error: unknown operation in user data handler");'
          // however this code part is reached often (maybe a xerces bug). Hence do nothing here.
}

const string DOMParser::domParserKey("http://openmbv.berlios.de/MBXMLUtils/domParser");
DOMParserUserDataHandler DOMParser::userDataHandler;

shared_ptr<DOMParser> DOMParser::create(bool validate_) {
  shared_ptr<DOMParser> ret(new DOMParser(validate_));
  ret->me=ret;
  return ret;
}

DOMParser::DOMParser(bool validate_) : validate(validate_) {
  // get DOM implementation and create parser
  domImpl=DOMImplementationRegistry::getDOMImplementation(X()%"");
  parser.reset(domImpl->createLSParser(DOMImplementation::MODE_SYNCHRONOUS, XMLUni::fgDOMXMLSchemaType),
    bind(&DOMLSParser::release, _1));
  // convert parser to AbstractDOMParser and store in parser filter
  AbstractDOMParser *abstractParser=dynamic_cast<AbstractDOMParser*>(parser.get());
  if(!abstractParser)
    throw runtime_error("Internal error: Parser is not of type AbstractDOMParser.");
  locationFilter.setParser(this);
  parser->setFilter(&locationFilter);
  // entity resolver (do never load entities using network)
  abstractParser->setDisableDefaultEntityResolution(true);
  // configure the parser
  DOMConfiguration *config=parser->getDomConfig();
  config->setParameter(XMLUni::fgDOMErrorHandler, &errorHandler);
  config->setParameter(XMLUni::fgXercesUserAdoptsDOMDocument, true);
  config->setParameter(XMLUni::fgXercesDoXInclude, false);
  //config->setParameter(XMLUni::fgDOMCDATASections, false); // is not implemented by xercesc handle it by DOMParser
  // configure validating parser
  if(validate) {
    config->setParameter(XMLUni::fgXercesScannerName, XMLUni::fgSGXMLScanner);
    config->setParameter(XMLUni::fgDOMValidate, true);
    config->setParameter(XMLUni::fgXercesSchema, true);
    config->setParameter(XMLUni::fgXercesUseCachedGrammarInParse, true);
    config->setParameter(XMLUni::fgXercesDOMHasPSVIInfo, true);
    typeDerHandler.setParser(this);
    abstractParser->setPSVIHandler(&typeDerHandler);
    loadGrammar(getInstallPath()/"share"/"mbxmlutils"/"schema"/"http___www_w3_org"/"XInclude.xsd");
  }
}

void DOMParser::loadGrammar(const path &schemaFilename) {
  // callable only for validating parsers
  if(!validate)
    throw runtime_error("Loading a XML schema in a none validating parser is not possible.");
  // load grammar
  errorHandler.resetCounter();
  parser->loadGrammar(X()%schemaFilename.string(CODECVT), Grammar::SchemaGrammarType, true);
  if(errorHandler.getNumErrors()>0)
    throw runtime_error(str(format("Loading XML schema failed: %1% Errors, %2% Warnings, see above.")%
      errorHandler.getNumErrors()%errorHandler.getNumWarnings()));
}

void DOMParser::handleXIncludeAndCDATA(DOMElement *&e) {
  // handle xinclude
  if(E(e)->getTagName()==XINCLUDE%"include") {
    path incFile=E(e)->convertPath(E(e)->getAttribute("href"));
    boost::shared_ptr<xercesc::DOMDocument> incDoc=parse(incFile);
    E(incDoc->getDocumentElement())->workaroundDefaultAttributesOnImportNode();// workaround
    DOMNode *incNode=e->getOwnerDocument()->importNode(incDoc->getDocumentElement(), true);
    e->getParentNode()->replaceChild(incNode, e)->release();
    e=static_cast<DOMElement*>(incNode);
    return;
  }
  // combine CDATA and text nodes
  for(DOMNode *c=e->getFirstChild(); c!=0; c=c->getNextSibling())
    if(c->getNodeType()==DOMNode::TEXT_NODE || c->getNodeType()==DOMNode::CDATA_SECTION_NODE) {
      DOMText *replace=static_cast<DOMText*>(e->insertBefore(e->getOwnerDocument()->createTextNode(X()%""), c));
      string data;
      while(c && (c->getNodeType()==DOMNode::TEXT_NODE || c->getNodeType()==DOMNode::CDATA_SECTION_NODE)) {
        data+=X()%static_cast<DOMText*>(c)->getData();
        DOMNode *del=c;
        c=c->getNextSibling();
        e->removeChild(del)->release();
      }
      replace->setData(X()%data);
      break;
    }
  // walk tree
  for(DOMElement *c=e->getFirstElementChild(); c!=0; c=c->getNextElementSibling()) {
    handleXIncludeAndCDATA(c);
    if(c==NULL) break;
  }
}

shared_ptr<DOMDocument> DOMParser::parse(const path &inputSource) {
  if(!exists(inputSource))
    throw runtime_error("XML document "+inputSource.string(CODECVT)+" not found");
  // reset error handler and parser document and throw on errors
  errorHandler.resetCounter();
  shared_ptr<DOMDocument> doc(parser->parseURI(X()%inputSource.string(CODECVT)), bind(&DOMDocument::release, _1));
  if(errorHandler.getNumErrors()>0)
    throw runtime_error(str(format("Validation failed: %1% Errors, %2% Warnings, see above.")%
      errorHandler.getNumErrors()%errorHandler.getNumWarnings()));
  // set file name
  DOMElement *root=doc->getDocumentElement();
  if(!E(root)->getFirstProcessingInstructionChildNamed("OriginalFilename")) {
    DOMProcessingInstruction *filenamePI=doc->createProcessingInstruction(X()%"OriginalFilename",
      X()%inputSource.string(CODECVT));
    root->insertBefore(filenamePI, root->getFirstChild());
  }
  // handle CDATA nodes
  handleXIncludeAndCDATA(root);
  // add a new shared_ptr<DOMParser> to document user data to extend the lifetime to the lifetime of all documents
  doc->setUserData(X()%domParserKey, new shared_ptr<DOMParser>(me), &userDataHandler);
  // return DOM document
  return doc;
}

void DOMParser::serialize(DOMNode *n, const path &outputSource, bool prettyPrint) {
  if(n->getNodeType()==DOMNode::DOCUMENT_NODE)
    static_cast<DOMDocument*>(n)->normalizeDocument();
  else
    n->getOwnerDocument()->normalizeDocument();

  DOMImplementation *impl=DOMImplementationRegistry::getDOMImplementation(X()%"");
  shared_ptr<DOMLSSerializer> ser(impl->createLSSerializer(), bind(&DOMLSSerializer::release, _1));
  ser->getDomConfig()->setParameter(XMLUni::fgDOMWRTFormatPrettyPrint, prettyPrint);
  if(!ser->writeToURI(n, X()%outputSource.string(CODECVT)))
    throw runtime_error("Serializing the document failed.");
}

void DOMParser::resetCachedGrammarPool() {
  parser->resetCachedGrammarPool();
  typeMap.clear();
}

shared_ptr<DOMDocument> DOMParser::createDocument() {
  shared_ptr<DOMDocument> doc(domImpl->createDocument(), bind(&DOMDocument::release, _1));
  doc->setUserData(X()%domParserKey, new shared_ptr<DOMParser>(me), &userDataHandler);
  return doc;
}

}
