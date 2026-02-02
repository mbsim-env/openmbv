#include <config.h>
#include "mbxmlutils/preprocess.h"
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMAttr.hpp>
#include <xercesc/dom/DOMProcessingInstruction.hpp>
#include <xercesc/dom/DOMTypeInfo.hpp>
#include <chrono>
#include <boost/scope_exit.hpp>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;
using namespace boost::filesystem;
using namespace fmatvec;

namespace MBXMLUtils {

const FQN Preprocess::embedFileNotFound = PV%"NoneExistentEmbedHref";

Preprocess::Preprocess(const path &inputFile, // a filename of a XML file used as input OR
                       variant<
                         shared_ptr<DOMParser>, // a direct parser OR
                         DOMElement*, // the root element of a DOM tree of a XML catalog file to create a parser OR
                         path // a filename of a XML catalog file to create a parser
                       > parserVariant,
                       bool trackDependencies
                      ) {
  if(MBXMLUtils::DOMEvalException::isHTMLOutputEnabled())
    msgStatic(Info)<<disableEscaping<<"<a name=\"MBXMLUTILS_PREPROCESS_CTOR\"></a>"<<enableEscaping;
  auto parser = initDependenciesAndParser(std::move(parserVariant), trackDependencies);
  document = parseCached(parser, inputFile, "XML input file.");
  msgStatic(Debug)<<"Finished: XML input file"<<endl;
  extractEvaluator();
}

Preprocess::Preprocess(istream &inputStream, // the input stream containing the XML file used as input
                       variant<
                         shared_ptr<DOMParser>, // a direct parser OR
                         DOMElement*, // the root element of a DOM tree of a XML catalog file to create a parser OR
                         path // a filename of a XML catalog file to create a parser
                       > parserVariant,
                       bool trackDependencies
                      ) {
  if(MBXMLUtils::DOMEvalException::isHTMLOutputEnabled())
    msgStatic(Info)<<disableEscaping<<"<a name=\"MBXMLUTILS_PREPROCESS_CTOR\"></a>"<<enableEscaping;
  auto parser = initDependenciesAndParser(std::move(parserVariant), trackDependencies);
  msgStatic(Info)<<"Load, parse and validate input stream."<<endl;
  document = parseCached(parser, inputStream, "XML input file.");
  msgStatic(Debug)<<"Finished: XML input file"<<endl;
  extractEvaluator();
}

std::shared_ptr<DOMParser> Preprocess::initDependenciesAndParser(std::variant<
                                         std::shared_ptr<MBXMLUtils::DOMParser>, // a direct parser OR
                                         xercesc::DOMElement*, // the root element of a DOM tree of a XML catalog file to create a parser OR
                                         boost::filesystem::path // a filename of a XML catalog file to create a parser
                                       > parserVariant,
                                       bool trackDependencies) {
  if(trackDependencies)
    dependencies = make_unique<std::vector<boost::filesystem::path>>();

  if(const path* xmlCatalogFile = get_if<path>(&parserVariant)) {
    msgStatic(Debug)<<"Create a validating XML parser from XML catalog file."<<endl;
    parserVariant = DOMParser::create(*xmlCatalogFile);
  }
  else if(DOMElement*const* xmlCatalogEle = get_if<DOMElement*>(&parserVariant)) {
    msgStatic(Debug)<<"Create a validating XML parser from XML catalog element."<<endl;
    parserVariant = DOMParser::create(*xmlCatalogEle);
  }
  return get<shared_ptr<DOMParser>>(parserVariant);
}

Preprocess::Preprocess(const shared_ptr<DOMDocument> &inputDoc, bool trackDependencies) {
  if(MBXMLUtils::DOMEvalException::isHTMLOutputEnabled())
    msgStatic(Info)<<disableEscaping<<"<a name=\"MBXMLUTILS_PREPROCESS_CTOR\"></a>"<<enableEscaping;
  if(trackDependencies)
    dependencies = make_unique<std::vector<boost::filesystem::path>>();

  document = inputDoc;
  msgStatic(Info)<<"Validate document."<<endl;
  D(document)->validate();
  msgStatic(Debug)<<"Finished: Validate document."<<endl;
  extractEvaluator();
}

Preprocess::~Preprocess() {
  if(MBXMLUtils::DOMEvalException::isHTMLOutputEnabled())
    msgStatic(Info)<<disableEscaping<<"<a name=\"MBXMLUTILS_PREPROCESS_DTOR\"></a>"<<enableEscaping;
}

const vector<path>& Preprocess::getDependencies() const {
  if(!preprocessed)
    throw DOMEvalException("Preprocess::getDependencies() is only useful after Preprocess::processAndGetDocument!", document->getDocumentElement());
  if(!dependencies)
    throw DOMEvalException("Preprocess::getDependencies() is only valid if trackDependencies=true was used in the constructor!", document->getDocumentElement());
  return *dependencies;
}

shared_ptr<Eval> Preprocess::getEvaluator() const {
  return eval;
}

void Preprocess::setParam(const shared_ptr<ParamSet>& param_) {
  param = param_;
}

shared_ptr<Preprocess::ParamSet> Preprocess::getParam() const {
  return param;
}

void Preprocess::extractEvaluator() {
  // check if the root element is valid -> if its a Embed -> no counterName, count or onlyif allowed
  auto root=document->getDocumentElement();
  if(E(root)->getTagName()==PV%"Embed") {
    if(E(root)->hasAttribute("counterName") || E(root)->hasAttribute("count") || E(root)->hasAttribute("href") ||
       (E(root)->hasAttribute("onlyif") && E(root)->getAttribute("onlyif")!="1"))
      throw DOMEvalException("A Embed element as root element is not allowed to have a counterName, count, onlyif or href attribute.", root);
  }

  // create a clean evaluator (get the evaluator name first form the dom)

  string evalName="octave"; // default evaluator
  DOMElement *evaluator;
  if(E(root)->getTagName()==PV%"Embed") {
    // if the root element IS A Embed than the <evaluator> element is the first child of the
    // first (none pv:Parameter) child of the root element
    auto r=root->getFirstElementChild();
    if(E(r)->getTagName()==PV%"Parameter")
      r=r->getNextElementSibling();
    evaluator=E(r)->getFirstElementChildNamed(PV%"evaluator");
  }
  else
    // if the root element IS NOT A Embed than the <evaluator> element is the first child root element
    evaluator=E(root)->getFirstElementChildNamed(PV%"evaluator");
  if(evaluator) {
    auto text=E(evaluator)->getText<string>();
    evalName=text;
  }

  eval = Eval::createEvaluator(evalName, dependencies.get());
}

shared_ptr<DOMDocument> Preprocess::processAndGetDocument() {
  if(MBXMLUtils::DOMEvalException::isHTMLOutputEnabled())
    msgStatic(Info)<<disableEscaping<<"<a name=\"MBXMLUTILS_PREPROCESS_START\"></a>"<<enableEscaping;
  msgStatic(Info)<<"Start XML preprocessing."<<endl;
  auto start = std::chrono::high_resolution_clock::now();
  if(preprocessed)
    throw DOMEvalException("Preprocess::processAndGetDocument and only be called ones!", document->getDocumentElement());

  // embed/validate/unit/eval files
  auto mainxmlele=document->getDocumentElement();
  if(!param)
    param = make_shared<ParamSet>();
  int dummy;
  Preprocess::preprocess(mainxmlele, dummy, param);
  // preprocess(...) may invalidate all DOMNode's except DOMDocument (due to revalidation by serialize/reparse).
  // Hence, get the mainxmlele (root DOMElement) again from the DOMDocument aver preprocess(...).
  mainxmlele=document->getDocumentElement();

  // adapt the evaluator in the dom (reset evaluator because it may change if the root element is a Embed)
  auto evaluator=E(mainxmlele)->getFirstElementChildNamed(PV%"evaluator");
  if(evaluator)
    E(evaluator)->getFirstTextChild()->setData(X()%"xmlflat");
  else {
    evaluator=D(document)->createElement(PV%"evaluator");
    evaluator->appendChild(document->createTextNode(X()%"xmlflat"));
    mainxmlele->insertBefore(evaluator, mainxmlele->getFirstChild());
  }

  preprocessed = true;

  auto end = std::chrono::high_resolution_clock::now();
  msgStatic(Info)<<"Finished XML preprocessing in "<<std::chrono::duration<double>(end-start).count()<<" seconds."<<endl;

#ifndef NDEBUG
  // in debug build check if Embed elements are still left, if so, its a programming bug!
  function<void(DOMElement*)> searchEmbed = [&searchEmbed](DOMElement *e) {
    if(E(e)->getTagName()==PV%"Embed")
      throw DOMEvalException("Internal error: the preprocessed XML file still contains a Embed element!", e);
    for(auto c=e->getFirstElementChild(); c!=nullptr; c=c->getNextElementSibling())
      searchEmbed(c);
  };
  searchEmbed(document->getDocumentElement());
#endif

  if(MBXMLUtils::DOMEvalException::isHTMLOutputEnabled())
    msgStatic(Info)<<disableEscaping<<"<a name=\"MBXMLUTILS_PREPROCESS_END\"></a>"<<enableEscaping;
  return document;
}

bool Preprocess::preprocess(DOMElement *&e, int &nrElementsEmbeded, const shared_ptr<ParamSet>& param, int embedXPathCount) {
  checkInterrupt();
  if(E(e)->getTagName()==PV%"Embed") {
    // handle the Embed element

    nrElementsEmbeded = 0;

    // check if only href OR child element (other than pv:Parameter) exists (This is not checked by the schema)
    DOMElement *inlineEmbedEle=e->getFirstElementChild();
    if(inlineEmbedEle && E(inlineEmbedEle)->getTagName()==PV%"Parameter")
      inlineEmbedEle=inlineEmbedEle->getNextElementSibling();
    if((inlineEmbedEle && E(e)->hasAttribute("href")) ||
       (!inlineEmbedEle && !E(e)->hasAttribute("href")))
      throw DOMEvalException("Only the href attribute OR a child element (expect pv:Parameter) is allowed in Embed!", e);

    // check that not both the parameterHref attribute and the child element pv:Parameter exists (This is not checked by the schema)
    DOMElement *inlineParamEle=e->getFirstElementChild();
    if(inlineParamEle && E(inlineParamEle)->getTagName()!=PV%"Parameter")
      inlineParamEle=nullptr;
    if(inlineParamEle && E(e)->hasAttribute("parameterHref"))
      throw DOMEvalException("Only the parameterHref attribute OR the child element pv:Parameter is allowed in Embed!", e);

    // check if attribute count AND counterName or none of both
    if((!E(e)->hasAttribute("count") &&  E(e)->hasAttribute("counterName")) ||
       ( E(e)->hasAttribute("count") && !E(e)->hasAttribute("counterName")))
      throw DOMEvalException("Only both, the count and counterName attribute must be given or none of both!", e);

    // we need the tag-name of the inlineEmbedEle later on, after revalidation (by serialize and reparse)
    // which will invalidate all DOMNode's except DOMDocument.
    // Hence, we save the tag-name now to avoid the need to access inlineEmbedEle later on (when it may be invalidated).
    FQN inlineEmbedEleTagName;
    if(inlineEmbedEle)
      inlineEmbedEleTagName = E(inlineEmbedEle)->getTagName();
  
    // evaluate count using parameters
    long count=1;
    if(E(e)->hasAttribute("count"))
      try { count=eval->cast<int>(eval->eval(E(e)->getAttributeNode("count"))); } RETHROW_AS_DOMEVALEXCEPTION(e)
  
    // counter name
    string counterName="MBXMLUtilsDummyCounterName";
    if(E(e)->hasAttribute("counterName"))
      counterName=E(e)->getAttribute("counterName");
  
    bool nodesInvalidated = false;
    XercesUniquePtr<DOMElement> localParamEle;
    path enewFilename;
    string enewInlineFilename;
    bool fileExists = false;
    path paramFile;
    XercesUniquePtr<DOMElement> enew;
    if(count>0) {
      // get element to embed and set file if this element is from href and save as enew
      if(E(e)->hasAttribute("href")) {
        // evaluate href
        string subst;
        try { subst=eval->cast<string>(eval->eval(E(e)->getAttributeNode("href"))); } RETHROW_AS_DOMEVALEXCEPTION(e)
        enewFilename=E(e)->convertPath(subst);
        if(dependencies)
          dependencies->push_back(enewFilename);

        // validate/load if file is given and save in enew
        fileExists = MBXMLUtils::exists(enewFilename);
        if(fileExists) {
          shared_ptr<DOMDocument> newdoc;
          try {
            newdoc=parseCached(D(document)->getParser(), enewFilename, "Embed file.", true);
          }
          catch(DOMEvalException& ex) {
            ex.appendContext(e),
            throw;
          }
          enew.reset(static_cast<DOMElement*>(e->getOwnerDocument()->importNode(newdoc->getDocumentElement(), true)));
        }
        else
          enew.reset(D(e->getOwnerDocument())->createElement(embedFileNotFound));
      }
      else
        // take the child element (inlineEmbedEle)
        enew.reset(static_cast<DOMElement*>(e->removeChild(inlineEmbedEle)));

      // add the OriginalFilename if not already added
      if(!E(e)->getEmbedData("MBXMLUtils_OriginalFilename").empty())
        E(enew)->setOriginalFilename(E(e)->getOriginalFilename());
      // include a processing instruction with the line number of the original element
      E(enew)->setOriginalElementLineNumber(E(e)->getLineNumber());
      E(enew)->setEmbedXPathCount(embedXPathCount);

      if(E(enew)->getTagName()!=embedFileNotFound && !enew->getSchemaTypeInfo()->getTypeName()) {
        nodesInvalidated = true;
        // Validate enew! only needed if enew does not have any type information yet.
        // (a unvalidated parse may happend when the root element was not known; mainly since the root element is a local element)
        //
        // The any-child element of the Embed element is of processContents="skip" but we need the type information
        // of all element in enew. We do this by:
        // 1. replacing the Embed element with enew
        // 2. revalidate the complete document
        // 3. revert the change of 1.
        // Very special care must be taken since 2. is done by serializing and reparsing the document which will
        // invalidate all DOMNode's except the DOMDocument.
        // Hence, we save the xpath of enew and get the enew element again in 3. using the saved xpath.
        auto doc = e->getOwnerDocument();

        DOMEvalException msg("WARNING: Revalidate document "+D(doc)->getDocumentFilename().string()+
                             " to populate this local element with type information.", enew.get());
        msgStatic(Debug)<<disableEscaping<<msg.what()<<enableEscaping<<endl;

        vector<int> xPathenew;
        XercesUniquePtr<DOMElement> savede;
        {
          auto p = e->getParentNode();
          auto enewPtr = enew.release();
          savede.reset(static_cast<DOMElement*>(p->replaceChild(enewPtr, e)));
          xPathenew = E(enewPtr)->getElementLocation();
        }
        auto oldRoot = D(doc)->validate(); // prevent release of the old root element until end of scope (we need "e" next)
        {
          auto enewPtr = static_cast<DOMElement*>(D(doc)->locateElement(xPathenew));
          auto p = enewPtr->getParentNode();
          enew.reset(static_cast<DOMElement*>(p->replaceChild(savede.release(), enewPtr)));
        }
      }

      // get element of the parameters of this embed and set paramFile if this element is from parameterHref and save as localParamEle
      if(E(e)->hasAttribute("parameterHref")) {
        // parameter from parameterHref attribute
        Eval::Value ret;
        try { ret=eval->eval(E(e)->getAttributeNode("parameterHref")); } RETHROW_AS_DOMEVALEXCEPTION(e)
        string subst;
        try { subst=eval->cast<string>(ret); } RETHROW_AS_DOMEVALEXCEPTION(e)
        paramFile=E(e)->convertPath(subst);
        if(MBXMLUtils::exists(paramFile)) {
          // add local parameter file to dependencies
          if(dependencies)
            dependencies->push_back(paramFile);
          // validate and local parameter file
          shared_ptr<DOMDocument> localparamxmldoc;
          try {
            localparamxmldoc=parseCached(D(document)->getParser(), paramFile, "Local parameter file.");
          }
          catch(DOMEvalException& ex) {
            ex.appendContext(e),
            throw;
          }
          if(E(localparamxmldoc->getDocumentElement())->getTagName()!=PV%"Parameter")
            throw DOMEvalException("The root element of a parameter file '"+paramFile.string()+"' must be {"+PV.getNamespaceURI()+"}Parameter", e);
          // generate local parameters
          localParamEle.reset(static_cast<DOMElement*>(e->getOwnerDocument()->importNode(localparamxmldoc->getDocumentElement(), true)));
          E(localParamEle)->setOriginalFilename(E(localparamxmldoc->getDocumentElement())->getOriginalFilename());
        }
        else
          localParamEle.reset(D(e->getOwnerDocument())->createElement(embedFileNotFound));
      }
      else if(inlineParamEle) {
        // inline parameter
        localParamEle.reset(static_cast<DOMElement*>(e->removeChild(inlineParamEle)));
      }

      if(localParamEle) {
        // include a processing instruction with the line number of the original element
        E(localParamEle)->setOriginalElementLineNumber(E(e)->getLineNumber());
        E(localParamEle)->setEmbedXPathCount(embedXPathCount);
      }

      // overwrite local parameter to/by param (only handle root level parameters)
      if(localParamEle && param && e->getParentNode()->getNodeType()==DOMNode::DOCUMENT_NODE) {

        // override parameters
        for(auto &[parName, parValue] : *param) {
          // search for a parameter named parName in localParamEle
          bool found=false;
          for(DOMElement *p=localParamEle->getFirstElementChild(); p!=nullptr; p=p->getNextElementSibling()) {
            if(E(p)->getAttribute("name")==parName) {
              // if found overwrite this parameter
              try {
                eval->checkIfValueMatchesElement(parValue, p);
              }
              catch(const DOMEvalException &ex) {
                throw DOMEvalException("The parameter '"+parName+"' should be overwritten but its type does not match:\n"+ex.getMessage(), p);
              }
              Eval::setValue(p, parValue);
              msgStatic(Info)<<"Parameter '"<<parName<<"' overwritten with value "<<eval->cast<CodeString>(parValue)<<endl;
              found=true;
              break;
            }
          }
          if(!found)
            msgStatic(Warn)<<"Parameter '"<<parName<<"' not found and not overwritten"<<endl;
        }

        // output parameters to the caller
        param->clear();
        shared_ptr<Eval> plainEval=Eval::createEvaluator(eval->getName());
        for(DOMElement *p=localParamEle->getLastElementChild(); p!=nullptr; p=p->getPreviousElementSibling()) {
          auto name=E(p)->getAttribute("name");
          if(param->find(name)!=param->end())
            continue;
          Eval::Value parValue;
          // only add the parameter if it does not depend on others and is of type scalar, vector, matrix or string
          try {
            parValue=plainEval->eval(p);
            E(e)->removeAttribute("unit");
            E(e)->removeAttribute("convertUnit");
          }
          catch(exception &ex) {
            if(E(p)->getTagName()!=PV%"import")
              eval->msg(Warn)<<"The 'pv:"<<E(p)->getTagName().second<<"' parameter named '"
                             <<name<<"' is not provided as overwritable parameter. Cannot evaluate this parameter."<<endl;
            continue;
          }
          param->emplace(name, parValue);
        }
      }

      // generate a enewFilename for a inline embed element used for printing messages
      if(enewFilename.empty())
        enewInlineFilename=string("[inline element]:{").append(inlineEmbedEleTagName.first).append("}").append(inlineEmbedEleTagName.second);
    }

    // delete embed element and insert count time the new element
    DOMElement *embed=e;
    DOMNode *p=e->getParentNode();
    DOMElement *insertBefore=embed->getNextElementSibling();
    DOMEvalException embedEvalException("", embed);
    XercesUniquePtr<DOMElement> embedUniquePtr(static_cast<DOMElement*>(p->removeChild(embed)));
    int realCount=0;
    auto onlyifStr = E(embed)->getAttribute("onlyif");
    // embed by gets invalidated during the below for loop, hence we create already a DOMEvalException object with
    // the context of embed now for potential throwing it later on (the error message is set before)
    // This avoid that we need embed later on during the for loop.

    // the over the count!
    for(long i=1; i<=count; i++) {
      checkInterrupt();
      NewParamLevel newParamLevel(eval);
      Eval::Value ii=eval->create(static_cast<double>(i));
      eval->convertIndex(ii, false);
      eval->addParam(counterName, ii);
      eval->addParam(counterName+"_count", eval->create(static_cast<double>(count)));
      if(localParamEle && E(localParamEle)->getTagName()!=embedFileNotFound) {
        eval->msg(Info)<<"Generate local parameters for "<<(enewFilename.empty()?enewInlineFilename:(string("\"").append(enewFilename.string()).append("\"")))
                         <<" ("<<i<<"/"<<count<<")"<<endl;
        // before we call eval->addParamSet(...) we add localParamEle.get() to the DOM tree to propably handle the calling
        // context when a DOMEvalException is thrown. After the call we remove it again and store it again in localParamEle.
        auto localParamEleInDOM=static_cast<DOMElement*>(p->insertBefore(localParamEle.release(), nullptr));
        BOOST_SCOPE_EXIT(&localParamEle, &p, &localParamEleInDOM) {
          localParamEle.reset(static_cast<DOMElement*>(p->removeChild(localParamEleInDOM)));
        } BOOST_SCOPE_EXIT_END
        try { eval->addParamSet(localParamEleInDOM); } RETHROW_AS_DOMEVALEXCEPTION(e)
      }

      // embed only if 'onlyif' attribute is true
      bool onlyif=true;
      if(!onlyifStr.empty()) {
        try {
          onlyif=(eval->cast<double>(eval->eval(onlyifStr))==1);
        }
        catch(const exception &ex) {
          embedEvalException.setMessage(ex.what());
          throw embedEvalException;
        }
      }
      if(onlyif) {
        nrElementsEmbeded++;
        if(!enewFilename.empty() && !fileExists) {
          embedEvalException.setMessage(string("Embed file \"").append(enewFilename.string()).append("\" not found."));
          throw embedEvalException;
        }
        if(localParamEle && E(localParamEle)->getTagName()==embedFileNotFound) {
          embedEvalException.setMessage(string("Parameter file \"").append(paramFile.string()).append("\" not found."));
          throw embedEvalException;
        }
        realCount++;
        eval->msg(Info)<<"Embed "<<(enewFilename.empty()?enewInlineFilename:(string("\"").append(enewFilename.string()).append("\"")))<<" ("<<i<<"/"<<count<<")"<<endl;
        if(p->getNodeType()==DOMElement::DOCUMENT_NODE && realCount!=1) {
          embedEvalException.setMessage("An Embed being the root XML node must expand to exactly one element.");
          throw embedEvalException;
        }
        e=static_cast<DOMElement*>(p->insertBefore(
          // avoid cloneNode if we know that this is the last time we add enew
          i==count ? enew.release() : static_cast<DOMElement*>(enew->cloneNode(true)),
          insertBefore
        ));
        // include a processing instruction with the count number
        E(e)->setEmbedCountNumber(i);
    
        // apply embed to new element
        // (do not pass param here since we report only top level parameter sets)
        //
        // we need to store the location of the insertBefore and the p element since we may need to recover these elements
        // if "preprocess(...)" of this embed element invalidates nodes.
        vector<int> location;
        if(insertBefore)
          // if insertBefore != null we store its location and p (being the parent of insertBefore) is the parent element of this location
          location = E(insertBefore)->getElementLocation();
        else {
          // if insertBefore == null we store the location of p (being the parent of insertBefore)
          if(p->getNodeType()==DOMNode::ELEMENT_NODE) // p may be a document for which we cannot call getElementLocation but its location is a empty vector mfmf
            location = E(static_cast<DOMElement*>(p))->getElementLocation();
        }
        int dummy;
        bool ni = preprocess(e, dummy);
        nodesInvalidated = ni || nodesInvalidated;
        if(ni) { // if preprocess(e) has invalidated all nodes then restore p and insertBefore from the stored xpath
          if(insertBefore) {
            // location stores the location of insertBefore and p is the parent of insertBefore
            insertBefore=static_cast<DOMElement*>(D(document)->locateElement(location));
            p = insertBefore->getParentNode();
          }
          else {
            // location stores the location of p and insertBefore == null (beond the last child element of p)
            if(!location.empty()) // we need to handle the case the p is a DOMElement and p is a DOMDocument
              p=static_cast<DOMElement*>(D(document)->locateElement(location));
            else
              p=document.get();
          }
        }
      }
      else
        eval->msg(Info)<<"Skip embeding "<<(enewFilename.empty()?enewInlineFilename:(string("\"").append(enewFilename.string()).append("\"")))<<" ("<<i<<"/"<<count<<"); onlyif attribute is false"<<endl;
    }
    if(p->getNodeType()==DOMElement::DOCUMENT_NODE && realCount!=1) {
      embedEvalException.setMessage("An Embed being the root XML node must expand to exactly one element.");
      throw embedEvalException;
    }
    // no new element added, just the Embed was removed
    if(embed==e)
      e=nullptr;
    return nodesInvalidated;
  }
  else {
    // handle all elements other than the Embed element

    nrElementsEmbeded = -1;

    bool function=false;

    // evaluate attributes
    DOMNamedNodeMap *attr=e->getAttributes();
    for(int i=0; i<attr->getLength(); i++) {
      checkInterrupt();
      auto *a=static_cast<DOMAttr*>(attr->item(i));
      // skip xml* attributes
      if((X()%a->getName()).substr(0, 3)=="xml")
        continue;
      // check for functions
      if(A(a)->isDerivedFrom(PV%"symbolicFunctionArgNameType"))
        function=true;
      // skip attributes which are not evaluated
      if(!A(a)->isDerivedFrom(PV%"fullEval") && !A(a)->isDerivedFrom(PV%"partialEval"))
        continue;
      Eval::Value value;
      try { value=eval->eval(a); } RETHROW_AS_DOMEVALEXCEPTION(e)
      string s;
      try {
        if(eval->valueIsOfType(value, Eval::ScalarType)) {
          double d=eval->cast<double>(value);
          int i;
          if(tryDouble2Int(d, i))
            s=toString(i);
          else
            s=toString(d);
        }
        else if(eval->valueIsOfType(value, Eval::StringType))
          s=eval->cast<string>(value);
        else
          throw DOMEvalException("Attribute evaluations can only be of type scalar or string.", a);
      } RETHROW_AS_DOMEVALEXCEPTION(e)

      // attributes of type qnamePartialEval need special handling
      if(A(a)->isDerivedFrom(PV%"qnamePartialEval")) {
        // if defined by the [<nsuri>]<localname> syntax it must be converted to the syntax <prefix>:<localname>
        if(s.length()>0 && s[0]=='[') {
          size_t c=s.find(']');
          if(c==string::npos)
            throw DOMEvalException("QName attribute value defined by '[uri]localname' syntax but no ] found.", a);
          E(e)->setAttribute(X()%a->getName(), FQN(s.substr(1,c-1), s.substr(c+1)));
        }
        else
          a->setValue(X()%s);
      }
      // all attributes of type other types just set the value
      else
        a->setValue(X()%s);
    }

    // evaluate element if it must be evaluated
    if(E(e)->isDerivedFrom(PV%"scalar") ||
       E(e)->isDerivedFrom(PV%"vector") ||
       E(e)->isDerivedFrom(PV%"matrix") ||
       E(e)->isDerivedFrom(PV%"fullEval") ||
       E(e)->isDerivedFrom(PV%"integerVector") ||
       E(e)->isDerivedFrom(PV%"indexVector") ||
       E(e)->isDerivedFrom(PV%"indexMatrix") ||
       function) {
      Eval::Value value;
      try { value=eval->eval(e); } RETHROW_AS_DOMEVALEXCEPTION(e)
      E(e)->removeAttribute("unit");
      E(e)->removeAttribute("convertUnit");
      // remove all child elements and child text nodes (since we add the evaluated value as the data = text node)
      DOMNode *nNext;
      for(auto *n=e->getFirstChild(); n; n=nNext) {
        nNext = n->getNextSibling();
        if(n->getNodeType()==DOMNode::ELEMENT_NODE || n->getNodeType()==DOMNode::TEXT_NODE || n->getNodeType()==DOMNode::CDATA_SECTION_NODE)
          e->removeChild(n)->release();
      }
      DOMNode *node;
      DOMDocument *doc=e->getOwnerDocument();
      try {
        node=doc->createTextNode(X()%eval->cast<CodeString>(value));
      } RETHROW_AS_DOMEVALEXCEPTION(e)
      e->appendChild(node);
    }

    // handle elements of type PV%"script"
    if(E(e)->isDerivedFrom(PV%"script")) {
      // eval element: for PV%"script" a string containing all parameters in xmlflateval notation is returned.
      Eval::Value value;
      try { value=eval->eval(e); } RETHROW_AS_DOMEVALEXCEPTION(e)
      // add processing instruction <?ScriptParameter ...?>
      // add processing instruction <?ScriptParameter ...?>
      E(e)->addProcessingInstructionChildNamed("ScriptParameter", eval->cast<string>(value));
    }
  }
  
  // walk tree
  DOMElement *c=e->getFirstElementChild();
  int childEmbedXPathCount=0;
  bool nodesInvalidated = false;
  while(c) {
    // pass param and the new parent XPath to preprocess
    DOMElement *n=c->getNextElementSibling();
    // the call to preprocess(...) may invalidate all DOMNote's due to revalidation
    // using serialize and reparse.
    // Hence, we save the xpath of n and get n after calling preprocess(...) again
    // using this save xpath.
    vector<int> xpathn;
    if(n)
      xpathn = E(n)->getElementLocation();
    if(E(c)->getTagName()==PV%"Embed") childEmbedXPathCount++;
    int nrElementsEmbeded;
    bool ni = preprocess(c, nrElementsEmbeded, shared_ptr<ParamSet>(), childEmbedXPathCount);
    // if c is a Embed element we need to fix xpathn since the count from the parent element of c to the next sibling element of c
    // is wrong in xpathn if the Embed element is replaced by no element (e.g. Embed-count=0) or by more then one element (e.g. Embed-count>1).
    // This is needed to ensure the locateElement in the below code finds the correct next sibling element of c if we need to use locateElement
    // because all node pointers were invalidated (due to a reparse/revalidate).
    if(!xpathn.empty() && nrElementsEmbeded != -1)
      xpathn[0]+=nrElementsEmbeded-1;
    nodesInvalidated = ni || nodesInvalidated;
    if(nodesInvalidated) {
      // this will work always but is slower then c=n
      if(n)
        c=static_cast<DOMElement*>(D(document)->locateElement(xpathn));
      else
        c=nullptr;
    }
    else
      c=n;
  }
  return nodesInvalidated;
}

shared_ptr<DOMDocument> Preprocess::parseCached(const shared_ptr<DOMParser> &parser, const path &inputFile,
                                                const string &msg, bool allowUnknownRootElement) {
  auto [it, insert] = parsedFiles.emplace(absolute(inputFile), shared_ptr<DOMDocument>());
  if(!insert) {
    msgStatic(Info)<<"Reuse cached file "<<inputFile<<": "<<msg<<endl;
    return it->second;
  }
  msgStatic(Info)<<"Load, parse and validate file "<<inputFile<<": "<<msg<<endl;
  shared_ptr<DOMDocument> doc;
  if(allowUnknownRootElement) {
    try {
      doc = parser->parse(inputFile, dependencies.get(), false);
    }
    catch(const DOMEvalException &ex) {
      if(ex.getNodeType()!=DOMNode::DOCUMENT_NODE) // if anything except the root element caused the error -> throw
                                                   // else return a unvalidated document
        throw;
      // on error parse without validation
      if(!noneValidatingParser)
        noneValidatingParser = DOMParser::create();
      doc = noneValidatingParser->parse(inputFile, dependencies.get(), false);
    }
  }
  else
    doc = parser->parse(inputFile, dependencies.get(), false);
  return it->second = doc;
}

shared_ptr<DOMDocument> Preprocess::parseCached(const shared_ptr<DOMParser> &parser, istream &inputStream,
                                                const string &msg, bool allowUnknownRootElement) {
  // read the entire stream (we need the content at least two times)
  std::stringstream buffer;
  buffer<<inputStream.rdbuf();
  string inputString=buffer.str();
  auto hashNr=hash<string>{}(inputString);
  auto [it, insert] = parsedStream.emplace(hashNr, shared_ptr<DOMDocument>());
  if(!insert) {
    msgStatic(Info)<<"Reuse cached input stream (hash="<<hashNr<<"): "<<msg<<endl;
    return it->second;
  }
  msgStatic(Info)<<"Parse and validate input stream (hash="<<hashNr<<"): "<<msg<<endl;
  shared_ptr<DOMDocument> doc;
  if(allowUnknownRootElement) {
    try {
      stringstream str(inputString); // no std::move here since inputString may be needed a second time inside the catch
      doc = parser->parse(str, dependencies.get(), false);
    }
    catch(const DOMEvalException &ex) {
      if(ex.getNodeType()!=DOMNode::DOCUMENT_NODE) // if anything except the root element caused the error -> throw
                                                   // else return a unvalidated document
        throw;
      // on error parse without validation
      if(!noneValidatingParser)
        noneValidatingParser = DOMParser::create();
      stringstream str(std::move(inputString)); // this error will be done with c++20
      doc = noneValidatingParser->parse(str, dependencies.get(), false);
    }
  }
  else {
    stringstream str(std::move(inputString)); // this error will be done with c++20
    doc = parser->parse(str, dependencies.get(), false);
  }
  return it->second = doc;
}

}
