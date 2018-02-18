#include <config.h>
#include "mbxmlutils/preprocess.h"
#include "mbxmlutils/eval.h"
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMAttr.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMProcessingInstruction.hpp>
#include "mbxmlutilshelper/casadiXML.h"
#include <fmatvec/toString.h>
#include <boost/scope_exit.hpp>

using namespace std;
using namespace std::placeholders;
using namespace MBXMLUtils;
using namespace xercesc;
using namespace boost::filesystem;

namespace MBXMLUtils {

void Preprocess::preprocess(const shared_ptr<DOMParser>& parser, const shared_ptr<Eval> &eval, vector<path> &dependencies, DOMElement *&e,
                            const shared_ptr<XPathParamSet>& param, const string &parentXPath, int embedXPathCount,
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
        Eval::Value ret=eval->eval(E(e)->getAttributeNode("href"), e);
        string subst;
        try {
          if(eval->valueIsOfType(ret, Eval::ScalarType))
            try {
              subst=fmatvec::toString(eval->cast<int>(ret));
            }
            catch(const DOMEvalException&) {
              subst=fmatvec::toString(eval->cast<double>(ret));
            }
          else if(eval->valueIsOfType(ret, Eval::StringType))
            subst=eval->cast<string>(ret);
          else
            throw DOMEvalException("Attribute evaluations can only be of type scalar or string.", e, E(e)->getAttributeNode("href"));
        } RETHROW_MBXMLUTILS(e)
        file=E(e)->convertPath(subst);
        dependencies.push_back(file);
      }
    
      // evaluate count using parameters
      long count=1;
      if(E(e)->hasAttribute("count"))
        try { count=eval->cast<int>(eval->eval(E(e)->getAttributeNode("count"), e)); } RETHROW_MBXMLUTILS(e)
    
      // couter name
      string counterName="MBXMLUtilsDummyCounterName";
      if(E(e)->hasAttribute("counterName"))
        try { counterName=eval->cast<string>(eval->eval(E(e)->getAttributeNode("counterName"), e)); } RETHROW_MBXMLUTILS(e)
    
      shared_ptr<DOMElement> enew;
      // validate/load if file is given
      if(!file.empty()) {
        eval->msg(Info)<<"Read and validate "<<file<<endl;
        shared_ptr<DOMDocument> newdoc=parser->parse(file, &dependencies, false);
        E(newdoc->getDocumentElement())->workaroundDefaultAttributesOnImportNode();// workaround
        enew.reset(static_cast<DOMElement*>(e->getOwnerDocument()->importNode(newdoc->getDocumentElement(), true)),
          bind(&DOMElement::release, _1));
      }
      else { // or take the child element (inlineEmbedEle)
        enew.reset(static_cast<DOMElement*>(e->removeChild(inlineEmbedEle)),
          bind(&DOMElement::release, _1));
      }

      // set the XPath of this (Embed) element to the name of the target Embed element (including the proper position)
      int pos=++(*position)[E(enew)->getTagName()];
      thisXPath="{"+E(enew)->getTagName().first+"}"+E(enew)->getTagName().second+"["+fmatvec::toString(pos)+"]";
    
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
      shared_ptr<DOMDocument> localparamxmldoc;
      if(inlineParamEle) { // inline parameter
        E(inlineParamEle)->setOriginalFilename();
        localParamEle.reset(static_cast<DOMElement*>(e->removeChild(inlineParamEle)), bind(&DOMElement::release, _1));
      }
      else if(E(e)->hasAttribute("parameterHref")) { // parameter from parameterHref attribute
        Eval::Value ret=eval->eval(E(e)->getAttributeNode("parameterHref"), e);
        string subst;
        try { subst=eval->cast<string>(ret); } RETHROW_MBXMLUTILS(e)
        path paramFile=E(e)->convertPath(subst);
        // add local parameter file to dependencies
        dependencies.push_back(paramFile);
        // validate and local parameter file
        eval->msg(Info)<<"Read and validate local parameter file "<<paramFile<<endl;
        localparamxmldoc=parser->parse(paramFile, &dependencies, false);
        // generate local parameters
        E(localparamxmldoc->getDocumentElement())->workaroundDefaultAttributesOnImportNode();// workaround
        localParamEle.reset(static_cast<DOMElement*>(e->getOwnerDocument()->importNode(localparamxmldoc->getDocumentElement(), true)),
          bind(&DOMElement::release, _1));
      }

      // add/overwrite local parameter to/by param
      if(localParamEle && param) {
        pair<XPathParamSet::iterator, bool> ret=param->insert(make_pair(parentXPath+"/"+thisXPath, ParamSet()));
        // no such XPath in param -> output this parameter set to the caller
        if(ret.second) {
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
              continue;
            }
            ret.first->second.emplace_back(E(p)->getAttribute("name"), parValue);
          }
        }
        // XPath already existing in param (specified by caller) -> overwrite all these parameters
        else {
          for(auto & it : ret.first->second) {
            // serach for a parameter named it->first in localParamEle
            for(DOMElement *p=localParamEle->getFirstElementChild(); p!=nullptr; p=p->getNextElementSibling()) {
              if(E(p)->getAttribute("name")==it.first) {
                // if found overwrite this parameter
                p->removeChild(E(p)->getFirstTextChild())->release();
                Eval::setValue(p, it.second);
              }
            }
          }
        }
      }

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
        if(localParamEle) {
          eval->msg(Info)<<"Generate local parameters for "<<(file.empty()?"<inline element>":file)
                           <<" ("<<i<<"/"<<count<<")"<<endl;
          eval->addParamSet(localParamEle.get());
        }

        // embed only if 'onlyif' attribute is true
        bool onlyif=true;
        if(E(embed)->hasAttribute("onlyif"))
          try { onlyif=(eval->cast<double>(eval->eval(E(embed)->getAttributeNode("onlyif"), embed))==1); } RETHROW_MBXMLUTILS(embed)
        if(onlyif) {
          realCount++;
          eval->msg(Info)<<"Embed "<<(file.empty()?"<inline element>":file)<<" ("<<i<<"/"<<count<<")"<<endl;
          if(p->getNodeType()==DOMElement::DOCUMENT_NODE && realCount!=1)
            throw DOMEvalException("An Embed being the root XML node must expand to exactly one element.", embed);
          e=static_cast<DOMElement*>(p->insertBefore(enew->cloneNode(true), insertBefore));
          // include a processing instruction with the count number
          E(e)->setEmbedCountNumber(i);
      
          // apply embed to new element
          // (do not pass param here since we report only top level parameter sets; parentXPath is also not longer needed)
          preprocess(parser, eval, dependencies, e);
        }
        else
          eval->msg(Info)<<"Skip embeding "<<(file.empty()?"<inline element>":file)<<" ("<<i<<"/"<<count<<"); onlyif attribute is false"<<endl;
      }
      if(p->getNodeType()==DOMElement::DOCUMENT_NODE && realCount!=1)
        throw DOMEvalException("An Embed being the root XML node must expand to exactly one element.", embed);
      // no new element added, just the Embed was removed
      if(embed==e)
        e=nullptr;
      return;
    }
    else if(E(e)->getTagName()==casadi::CASADI%"Function")
      return; // skip processing of Function elements
    else {
      bool isCasADi=false;

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
        // check for casadi functions
        if(A(a)->isDerivedFrom(PV%"symbolicFunctionArgNameType"))
          isCasADi=true;
        // skip attributes which are not evaluated
        if(!A(a)->isDerivedFrom(PV%"fullEval") && !A(a)->isDerivedFrom(PV%"partialEval"))
          continue;
        Eval::Value value=eval->eval(a, e);
        string s;
        try {
          if(eval->valueIsOfType(value, Eval::ScalarType))
            try {
              s=fmatvec::toString(eval->cast<int>(value));
            }
            catch(const DOMEvalException&) {
              s=fmatvec::toString(eval->cast<double>(value));
            }
          else if(eval->valueIsOfType(value, Eval::StringType))
            s=eval->cast<string>(value);
          else
            throw DOMEvalException("Attribute evaluations can only be of type scalar or string.", e, a);
        } RETHROW_MBXMLUTILS(e)

        // attributes of type qnamePartialEval need special handling
        if(A(a)->isDerivedFrom(PV%"qnamePartialEval")) {
          // if defined by the [<nsuri>]<localname> syntax it must be converted to the syntax <prefix>:<localname>
          if(s.length()>0 && s[0]=='[') {
            size_t c=s.find(']');
            if(c==string::npos)
              throw DOMEvalException("QName attribute value defined by [<uri>]<localname> syntax but no ] found.", e, a);
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
         isCasADi) {
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
          if(eval->valueIsOfType(value, Eval::FunctionType))
            node=eval->cast<DOMElement*>(value, doc);
          else
            node=doc->createTextNode(X()%eval->cast<CodeString>(value));
        } RETHROW_MBXMLUTILS(e)
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
      preprocess(parser, eval, dependencies, c, param, parentXPath+"/"+thisXPath, embedXPathCount);
      c=n;
    }
  } RETHROW_MBXMLUTILS(e);
}

}
