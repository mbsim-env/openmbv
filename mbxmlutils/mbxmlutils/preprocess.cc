#include <config.h>
#include "mbxmlutils/preprocess.h"
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMAttr.hpp>
#include <xercesc/dom/DOMProcessingInstruction.hpp>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;
using namespace boost::filesystem;

namespace MBXMLUtils {

Preprocess::Preprocess(const path &inputFile, // a filename of a XML file used as input OR
                       variant<
                         shared_ptr<DOMParser>, // a direct parser OR
                         DOMElement*, // the root element of a DOM tree of a XML catalog file to create a parser OR
                         path // a filename of a XML catalog file to create a parser
                       > parserVariant
                      ) {
  if(const path* xmlCatalogFile = get_if<path>(&parserVariant)) {
    fmatvec::Atom::msgStatic(fmatvec::Atom::Debug)<<"Create a validating XML parser from XML catalog file."<<endl;
    parserVariant = DOMParser::create(*xmlCatalogFile);
  }
  else if(DOMElement*const* xmlCatalogEle = get_if<DOMElement*>(&parserVariant)) {
    fmatvec::Atom::msgStatic(fmatvec::Atom::Debug)<<"Create a validating XML parser from XML catalog element."<<endl;
    parserVariant = DOMParser::create(*xmlCatalogEle);
  }
  auto parser = get<shared_ptr<DOMParser>>(parserVariant);
  document = parseCached(parser, inputFile, dependencies, "XML input file.");
  extractEvaluator();
}

Preprocess::Preprocess(const shared_ptr<DOMDocument> &inputDoc) {
  document = inputDoc;
  Atom::msgStatic(Atom::Info)<<"Validate document."<<endl;
  D(document)->validate();
  extractEvaluator();
}

const vector<path>& Preprocess::getDependencies() const {
  if(!preprocessed)
    throw DOMEvalException("Preprocess::getDependencies() is only useful after Preprocess::processAndGetDocument!", document->getDocumentElement());
  return dependencies;
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
  // create a clean evaluator (get the evaluator name first form the dom)

  string evalName="octave"; // default evaluator
  DOMElement *evaluator;
  if(E(document->getDocumentElement())->getTagName()==PV%"Embed") {
    // if the root element IS A Embed than the <evaluator> element is the first child of the
    // first (none pv:Parameter) child of the root element
    auto r=document->getDocumentElement()->getFirstElementChild();
    if(E(r)->getTagName()==PV%"Parameter")
      r=r->getNextElementSibling();
    evaluator=E(r)->getFirstElementChildNamed(PV%"evaluator");
  }
  else
    // if the root element IS NOT A Embed than the <evaluator> element is the first child root element
    evaluator=E(document->getDocumentElement())->getFirstElementChildNamed(PV%"evaluator");
  if(evaluator) {
    auto textEle=E(evaluator)->getFirstTextChild();
    auto text=textEle ? X()%textEle->getData() : "";
    evalName=text;
  }

  eval = Eval::createEvaluator(evalName, &dependencies);
}

shared_ptr<DOMDocument> Preprocess::processAndGetDocument() {
  if(preprocessed)
    throw DOMEvalException("Preprocess::processAndGetDocument and only be called ones!", document->getDocumentElement());

  // embed/validate/unit/eval files
  auto mainxmlele=document->getDocumentElement();
  if(!param)
    param = make_shared<ParamSet>();
  Preprocess::preprocess(mainxmlele, param);

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
  return document;
}

void Preprocess::preprocess(DOMElement *&e,
                            const shared_ptr<ParamSet>& param, const string &parentXPath, int embedXPathCount,
                            const shared_ptr<PositionMap>& position) {
  try {
    string thisXPath; // the XPath of this element (for a Embed its the target element name, for others its just the element name
    if(E(e)->getTagName()==PV%"Embed") {
      // check if only href OR child element (other than pv:Parameter) exists (This is not checked by the schema)
      DOMElement *inlineEmbedEle=e->getFirstElementChild();
      if(inlineEmbedEle && E(inlineEmbedEle)->getTagName()==PV%"Parameter")
        inlineEmbedEle=inlineEmbedEle->getNextElementSibling();
      if((inlineEmbedEle && E(e)->hasAttribute("href")) ||
         (!inlineEmbedEle && !E(e)->hasAttribute("href")))
        throw DOMEvalException("Only the href attribute OR a child element (expect pv:Parameter) is allowed in Embed!", e);
      // check if attribute count AND counterName or none of both
      if((!E(e)->hasAttribute("count") &&  E(e)->hasAttribute("counterName")) ||
         ( E(e)->hasAttribute("count") && !E(e)->hasAttribute("counterName")))
        throw DOMEvalException("Only both, the count and counterName attribute must be given or none of both!", e);
    
      // get file name if href attribute exist
      path file;
      if(E(e)->hasAttribute("href")) {
        Eval::Value ret=eval->eval(E(e)->getAttributeNode("href"));
        string subst;
        try {
          if(eval->valueIsOfType(ret, Eval::ScalarType)) {
            double d=eval->cast<double>(ret);
            int i;
            if(tryDouble2Int(d, i))
              subst=fmatvec::toString(i);
            else
              subst=fmatvec::toString(d);
          }
          else if(eval->valueIsOfType(ret, Eval::StringType))
            subst=eval->cast<string>(ret);
          else
            throw DOMEvalException("Attribute evaluations can only be of type scalar or string.", E(e)->getAttributeNode("href"));
        } RETHROW_AS_DOMEVALEXCEPTION(e)
        file=E(e)->convertPath(subst);
        dependencies.push_back(file);
      }
    
      // evaluate count using parameters
      long count=1;
      if(E(e)->hasAttribute("count"))
        try { count=eval->cast<int>(eval->eval(E(e)->getAttributeNode("count"))); } RETHROW_AS_DOMEVALEXCEPTION(e)
    
      // counter name
      string counterName="MBXMLUtilsDummyCounterName";
      if(E(e)->hasAttribute("counterName"))
        counterName=E(e)->getAttribute("counterName");
    
      shared_ptr<DOMElement> enew;
      // validate/load if file is given
      bool fileExists = false;
      if(!file.empty()) {
        fileExists = exists(file);
        if(fileExists) {
          shared_ptr<DOMDocument> newdoc;
          try {
            newdoc=parseCached(D(document)->getParser(), file, dependencies, "Embed file.");
          }
          catch(DOMEvalException& ex) {
            ex.appendContext(e),
            throw ex;
          }
          E(newdoc->getDocumentElement())->workaroundDefaultAttributesOnImportNode();// workaround
          enew.reset(static_cast<DOMElement*>(e->getOwnerDocument()->importNode(newdoc->getDocumentElement(), true)),
            [](auto && PH1) { if(PH1) PH1->release(); });
        }
        else
          enew.reset(D(e->getOwnerDocument())->createElement(PV%"NoneExistentEmbedHref"),
            [](auto && PH1) { if(PH1) PH1->release(); });
      }
      else { // or take the child element (inlineEmbedEle)
        enew.reset(static_cast<DOMElement*>(e->removeChild(inlineEmbedEle)),
          [](auto && PH1) { if(PH1) PH1->release(); });
      }

      // set the XPath of this (Embed) element to the name of the target Embed element (including the proper position)
      int pos=++(*position)[E(enew)->getTagName()];
      thisXPath="{"+E(enew)->getTagName().first+"}"+E(enew)->getTagName().second+"["+fmatvec::toString(pos)+"]";

      // check for direct successive Embed elements
      if(E(enew)->getTagName()==PV%"Embed")
        // if this deprecated feature is removed also fix embed.xsd
        eval->msg(Deprecated)<<"Direct successive pv:Embed elements are deprecated: "<<
          string(parentXPath).append("/").append(thisXPath)<<endl;
    
      // include a processing instruction with the line number of the original element
      E(enew)->setOriginalElementLineNumber(E(e)->getLineNumber());
      E(enew)->setEmbedXPathCount(embedXPathCount);
    
      // check that not both the parameterHref attribute and the child element pv:Parameter exists (This is not checked by the schema)
      DOMElement *inlineParamEle=e->getFirstElementChild();
      if(inlineParamEle && E(inlineParamEle)->getTagName()!=PV%"Parameter")
        inlineParamEle=nullptr;
      if(inlineParamEle && E(e)->hasAttribute("parameterHref"))
        throw DOMEvalException("Only the parameterHref attribute OR the child element pv:Parameter is allowed in Embed!", e);
      // get localParamEle
      shared_ptr<DOMElement> localParamEle;
      if(inlineParamEle) { // inline parameter
        E(inlineParamEle)->setOriginalFilename();
        localParamEle.reset(static_cast<DOMElement*>(e->removeChild(inlineParamEle)), [](auto && PH1) { if(PH1) PH1->release(); });
      }
      else if(E(e)->hasAttribute("parameterHref")) { // parameter from parameterHref attribute
        Eval::Value ret=eval->eval(E(e)->getAttributeNode("parameterHref"));
        string subst;
        try { subst=eval->cast<string>(ret); } RETHROW_AS_DOMEVALEXCEPTION(e)
        path paramFile=E(e)->convertPath(subst);
        // add local parameter file to dependencies
        dependencies.push_back(paramFile);
        // validate and local parameter file
        auto localparamxmldoc=parseCached(D(document)->getParser(), paramFile, dependencies, "Local parameter file.");
        if(E(localparamxmldoc->getDocumentElement())->getTagName()!=PV%"Parameter")
          throw DOMEvalException("The root element of a parameter file '"+paramFile.string()+"' must be {"+PV.getNamespaceURI()+"}Parameter", e);
        // generate local parameters
        E(localparamxmldoc->getDocumentElement())->workaroundDefaultAttributesOnImportNode();// workaround
        localParamEle.reset(static_cast<DOMElement*>(e->getOwnerDocument()->importNode(localparamxmldoc->getDocumentElement(), true)),
          [](auto && PH1) { if(PH1) PH1->release(); });
      }

      // overwrite local parameter to/by param (only handle root level parameters)
      if(localParamEle && param && parentXPath.empty()) {

        // override parameters
        for(auto & it : *param) {
          // serach for a parameter named it->first in localParamEle
          bool found=false;
          for(DOMElement *p=localParamEle->getFirstElementChild(); p!=nullptr; p=p->getNextElementSibling()) {
            if(E(p)->getAttribute("name")==it.first) {
              // if found overwrite this parameter
              p->removeChild(E(p)->getFirstTextChild())->release();
              Eval::setValue(p, it.second);
              fmatvec::Atom::msgStatic(fmatvec::Atom::Info)<<"Parameter '"<<it.first<<"' overwritten with value "<<eval->cast<CodeString>(it.second)<<endl;
              found=true;
              break;
            }
          }
          if(!found)
            fmatvec::Atom::msgStatic(fmatvec::Atom::Warn)<<"Parameter '"<<it.first<<"' not found and not overwritten"<<endl;
        }

        // output parameters to the caller
        param->clear();
        shared_ptr<Eval> plainEval=Eval::createEvaluator(eval->getName());
        for(DOMElement *p=localParamEle->getFirstElementChild(); p!=nullptr; p=p->getNextElementSibling()) {
          Eval::Value parValue;
          // only add the parameter if it does not depend on others and is of type scalar, vector, matrix or string
          try {
            parValue=plainEval->eval(p);
            E(e)->removeAttribute("unit");
            E(e)->removeAttribute("convertUnit");
          }
          catch(DOMEvalException &ex) {
            if(E(p)->getTagName()!=PV%"import")
              eval->msg(Warn)<<"The 'pv:"<<E(p)->getTagName().second<<"' parameter named '"
                             <<E(p)->getAttribute("name")<<"' is not provided as overwritable parameter. Cannot evaluate this parameter."<<endl;
            continue;
          }
          if(!param->emplace(E(p)->getAttribute("name"), parValue).second)
            throw DOMEvalException("Cannot add parameter. A parameter with the same name already exists.", p);
        }
      }

      string inlineElement;
      if(file.empty())
        inlineElement="[inline element]:{"+E(inlineEmbedEle)->getTagName().first+"}"+E(inlineEmbedEle)->getTagName().second;

      // delete embed element and insert count time the new element
      DOMElement *embed=e;
      DOMNode *p=e->getParentNode();
      DOMElement *insertBefore=embed->getNextElementSibling();
      p->removeChild(embed);
      BOOST_SCOPE_EXIT(&embed) {
        // release the embed element on scope exit
        embed->release();
      } BOOST_SCOPE_EXIT_END
      int realCount=0;
      for(long i=1; i<=count; i++) {
        NewParamLevel newParamLevel(eval);
        Eval::Value ii=eval->create(static_cast<double>(i));
        eval->convertIndex(ii, false);
        eval->addParam(counterName, ii);
        eval->addParam(counterName+"_count", eval->create(static_cast<double>(count)));
        if(localParamEle) {
          eval->msg(Info)<<"Generate local parameters for "<<(file.empty()?inlineElement:("\""+file.string()+"\""))
                           <<" ("<<i<<"/"<<count<<")"<<endl;
          eval->addParamSet(localParamEle.get());
        }

        // embed only if 'onlyif' attribute is true
        bool onlyif=true;
        if(E(embed)->hasAttribute("onlyif"))
          try { onlyif=(eval->cast<double>(eval->eval(E(embed)->getAttributeNode("onlyif")))==1); } RETHROW_AS_DOMEVALEXCEPTION(embed)
        if(onlyif) {
          if(!file.empty() && !fileExists)
            throw DOMEvalException("Embed file \""+file.string()+"\" not found.", embed);
          realCount++;
          eval->msg(Info)<<"Embed "<<(file.empty()?inlineElement:("\""+file.string()+"\""))<<" ("<<i<<"/"<<count<<")"<<endl;
          if(p->getNodeType()==DOMElement::DOCUMENT_NODE && realCount!=1)
            throw DOMEvalException("An Embed being the root XML node must expand to exactly one element.", embed);
          // this is needed to fix a Embed by href which itself uses a Embed by inline element
          if(E(e)->getFirstProcessingInstructionChildNamed("OriginalFilename"))
            E(enew)->setOriginalFilename(E(e)->getOriginalFilename());
          e=static_cast<DOMElement*>(p->insertBefore(enew->cloneNode(true), insertBefore));
          // include a processing instruction with the count number
          E(e)->setEmbedCountNumber(i);
      
          // apply embed to new element
          // (do not pass param here since we report only top level parameter sets; parentXPath is also not longer needed)
          preprocess(e);
        }
        else
          eval->msg(Info)<<"Skip embeding "<<(file.empty()?inlineElement:("\""+file.string()+"\""))<<" ("<<i<<"/"<<count<<"); onlyif attribute is false"<<endl;
      }
      if(p->getNodeType()==DOMElement::DOCUMENT_NODE && realCount!=1)
        throw DOMEvalException("An Embed being the root XML node must expand to exactly one element.", embed);
      // no new element added, just the Embed was removed
      if(embed==e)
        e=nullptr;
      return;
    }
    else {
      bool function=false;

      // set the XPath of this (none Embed) element to the name of the element itself (including the proper position)
      int pos=++(*position)[E(e)->getTagName()];
      thisXPath="{"+E(e)->getTagName().first+"}"+E(e)->getTagName().second+"["+fmatvec::toString(pos)+"]";

      // evaluate attributes
      DOMNamedNodeMap *attr=e->getAttributes();
      for(int i=0; i<attr->getLength(); i++) {
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
        Eval::Value value=eval->eval(a);
        string s;
        try {
          if(eval->valueIsOfType(value, Eval::ScalarType)) {
            double d=eval->cast<double>(value);
            int i;
            if(tryDouble2Int(d, i))
              s=fmatvec::toString(i);
            else
              s=fmatvec::toString(d);
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
        Eval::Value value=eval->eval(e);
        E(e)->removeAttribute("unit");
        E(e)->removeAttribute("convertUnit");
        // if a child element exists (xml*Group or fromFileGroup) then remove it
        if(e->getFirstElementChild())
          e->removeChild(e->getFirstElementChild())->release();
        // remove also all child text nodes
        while(E(e)->getFirstTextChild())
          e->removeChild(E(e)->getFirstTextChild())->release();
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
        Eval::Value value=eval->eval(e);
        // add processing instruction <?ScriptParameter ...?>
        DOMProcessingInstruction *scriptPar=e->getOwnerDocument()->createProcessingInstruction(X()%"ScriptParameter",
          X()%eval->cast<string>(value));
        e->insertBefore(scriptPar, e->getFirstChild());
      }
    }
    
    // walk tree
    DOMElement *c=e->getFirstElementChild();
    int embedXPathCount=0;
    while(c) {
      // pass param and the new parent XPath to preprocess
      DOMElement *n=c->getNextElementSibling();
      if(E(c)->getTagName()==PV%"Embed") embedXPathCount++;
      preprocess(c, shared_ptr<ParamSet>(), string(parentXPath).append("/").append(thisXPath), embedXPathCount);
      c=n;
    }
  } RETHROW_AS_DOMEVALEXCEPTION(e);
}

shared_ptr<DOMDocument> Preprocess::parseCached(const shared_ptr<DOMParser> &parser, const path &inputFile,
                                                vector<path> &dependencies, const std::string &msg) {
  auto [it, insert] = parsedFiles.emplace(absolute(inputFile), shared_ptr<DOMDocument>());
  if(!insert) {
    fmatvec::Atom::msgStatic(fmatvec::Atom::Info)<<"Reuse cached file "<<inputFile<<": "<<msg<<endl;
    return it->second;
  }
  fmatvec::Atom::msgStatic(fmatvec::Atom::Info)<<"Load, parse and validate file "<<inputFile<<": "<<msg<<endl;
  return it->second = parser->parse(inputFile, &dependencies, false);
}

}
