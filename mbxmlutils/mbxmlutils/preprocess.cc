#include <config.h>
#include "mbxmlutils/preprocess.h"
#include "mbxmlutils/octeval.h"
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMAttr.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <boost/lexical_cast.hpp>
#include "mbxmlutilshelper/casadiXML.h"

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;
using namespace boost;
using namespace boost::filesystem;

namespace MBXMLUtils {

void Preprocess::preprocess(shared_ptr<DOMParser> parser, Eval &eval, vector<path> &dependencies, DOMElement *&e,
                            shared_ptr<XPathParamSet> param, const string &parentXPath,
                            shared_ptr<PositionMap> position) {
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
        shared_ptr<void> ret=eval.eval(E(e)->getAttributeNode("href"), e);
        string subst;
        try {
          subst=eval.cast<string>(ret);
          if(eval.getType(ret)==Eval::StringType)
            subst=subst.substr(1, subst.length()-2);
        } MBXMLUTILS_RETHROW(e)
        file=E(e)->convertPath(subst);
        dependencies.push_back(file);
      }
    
      // evaluate count using parameters
      long count=1;
      if(E(e)->hasAttribute("count"))
        try { count=eval.cast<int>(eval.eval(E(e)->getAttributeNode("count"), e)); } MBXMLUTILS_RETHROW(e)
    
      // couter name
      string counterName="MBXMLUtilsDummyCounterName";
      if(E(e)->hasAttribute("counterName")) {
        try { counterName=eval.cast<string>(eval.eval(E(e)->getAttributeNode("counterName"), e)); } MBXMLUTILS_RETHROW(e)
        counterName=counterName.substr(1, counterName.length()-2);
      }
    
      shared_ptr<DOMElement> enew;
      // validate/load if file is given
      if(!file.empty()) {
        eval.msg(Info)<<"Read and validate "<<file<<endl;
        shared_ptr<DOMDocument> newdoc=parser->parse(file, &dependencies);
        E(newdoc->getDocumentElement())->workaroundDefaultAttributesOnImportNode();// workaround
        enew.reset(static_cast<DOMElement*>(e->getOwnerDocument()->importNode(newdoc->getDocumentElement(), true)),
          bind(&DOMElement::release, _1));
      }
      else { // or take the child element (inlineEmbedEle)
        E(inlineEmbedEle)->setOriginalFilename();
        enew.reset(static_cast<DOMElement*>(e->removeChild(inlineEmbedEle)),
          bind(&DOMElement::release, _1));
      }

      // set the XPath of this (Embed) element to the name of the target Embed element (including the proper position)
      int pos=++(*position)[E(enew)->getTagName()];
      thisXPath="{"+E(enew)->getTagName().first+"}"+E(enew)->getTagName().second+"["+lexical_cast<string>(pos)+"]";
    
      // include a processing instruction with the line number of the original element
      E(enew)->setOriginalElementLineNumber(E(e)->getLineNumber());
    
      // check that not both the parameterHref attribute and the child element pv:Parameter exists (This is not checked by the schema)
      DOMElement *inlineParamEle=e->getFirstElementChild();
      if(inlineParamEle && E(inlineParamEle)->getTagName()!=PV%"Parameter")
        inlineParamEle=NULL;
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
        shared_ptr<void> ret=eval.eval(E(e)->getAttributeNode("parameterHref"), e);
        string subst;
        try {
          subst=eval.cast<string>(ret);
          if(eval.getType(ret)==Eval::StringType)
            subst=subst.substr(1, subst.length()-2);
        } MBXMLUTILS_RETHROW(e)
        path paramFile=E(e)->convertPath(subst);
        // add local parameter file to dependencies
        dependencies.push_back(paramFile);
        // validate and local parameter file
        eval.msg(Info)<<"Read and validate local parameter file "<<paramFile<<endl;
        localparamxmldoc=parser->parse(paramFile, &dependencies);
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
          OctEval dummy;
          Eval &plainEval=dummy;
          for(const DOMElement *p=localParamEle->getFirstElementChild(); p!=NULL; p=p->getNextElementSibling()) {
            shared_ptr<void> parValue;
            // only add the parameter if it does not depend on others and is of type scalar, vector, matrix or string
            try {
              parValue=plainEval.eval(p);
            }
            catch(DOMEvalException &ex) {
              continue;
            }
            ret.first->second.push_back(make_pair(E(p)->getAttribute("name"), parValue));
          }
        }
        // XPath already existing in param (specified by caller) -> overwrite all these parameters
        else {
          for(ParamSet::iterator it=ret.first->second.begin(); it!=ret.first->second.end(); ++it) {
            // serach for a parameter named it->first in localParamEle
            for(DOMElement *p=localParamEle->getFirstElementChild(); p!=NULL; p=p->getNextElementSibling()) {
              if(E(p)->getAttribute("name")==it->first) {
                // if found overwrite this parameter
                p->removeChild(E(p)->getFirstTextChild())->release();
                p->appendChild(p->getOwnerDocument()->createTextNode(X()%eval.cast<string>(it->second)));
              }
            }
          }
        }
      }

      // delete embed element and insert count time the new element
      for(long i=1; i<=count; i++) {
        NewParamLevel newParamLevel(eval);
        eval.addParam(counterName, eval.create(static_cast<double>(i)));
        if(localParamEle) {
          eval.msg(Info)<<"Generate local parameters for "<<(file.empty()?"<inline element>":file)
                           <<" ("<<i<<"/"<<count<<")"<<endl;
          eval.addParamSet(localParamEle.get());
        }

        // embed only if 'onlyif' attribute is true
        bool onlyif=true;
        if(E(e)->hasAttribute("onlyif"))
          try { onlyif=(eval.cast<int>(eval.eval(E(e)->getAttributeNode("onlyif"), e))==1); } MBXMLUTILS_RETHROW(e)
        if(onlyif) {
          eval.msg(Info)<<"Embed "<<(file.empty()?"<inline element>":file)<<" ("<<i<<"/"<<count<<")"<<endl;
          DOMNode *p=e->getParentNode();
          if(i==1) {
            DOMElement *ereplaced=static_cast<DOMElement*>(p->insertBefore(enew->cloneNode(true), e->getNextSibling()));
            p->removeChild(e);
            e=ereplaced;
          }
          else
            e=static_cast<DOMElement*>(p->insertBefore(enew->cloneNode(true), e->getNextSibling()));
    
          // include a processing instruction with the count number
          E(e)->setEmbedCountNumber(i);
      
          // apply embed to new element
          // (do not pass param here since we report only top level parameter sets; parentXPath is also not longer needed)
          preprocess(parser, eval, dependencies, e);
        }
        else
          eval.msg(Info)<<"Skip embeding "<<(file.empty()?"<inline element>":file)<<" ("<<i<<"/"<<count<<"); onlyif attribute is false"<<endl;
      }
      return;
    }
    else if(E(e)->getTagName()==casadi::CASADI%"SXFunction")
      return; // skip processing of SXFunction elements
    else {
      bool isCasADi=false;

      // set the XPath of this (none Embed) element to the name of the element itself (including the proper position)
      int pos=++(*position)[E(e)->getTagName()];
      thisXPath="{"+E(e)->getTagName().first+"}"+E(e)->getTagName().second+"["+lexical_cast<string>(pos)+"]";

      // evaluate attributes
      DOMNamedNodeMap *attr=e->getAttributes();
      for(int i=0; i<attr->getLength(); i++) {
        DOMAttr *a=static_cast<DOMAttr*>(attr->item(i));
        // skip xml* attributes
        if((X()%a->getName()).substr(0, 3)=="xml")
          continue;
        // check for casadi functions
        if(A(a)->isDerivedFrom(PV%"symbolicFunctionArgNameType"))
          isCasADi=true;
        // skip attributes which are not evaluated
        if(!A(a)->isDerivedFrom(PV%"fullEval") && !A(a)->isDerivedFrom(PV%"partialEval"))
          continue;
        shared_ptr<void> value=eval.eval(a, e);
        string s;
        try {
          s=eval.cast<string>(value);
          if(eval.getType(value)==Eval::StringType)
            s=s.substr(1, s.length()-2);
        } MBXMLUTILS_RETHROW(e)
        a->setValue(X()%s);
      }

      // evaluate element if it must be evaluated
      if(E(e)->isDerivedFrom(PV%"scalar") ||
         E(e)->isDerivedFrom(PV%"vector") ||
         E(e)->isDerivedFrom(PV%"matrix") ||
         E(e)->isDerivedFrom(PV%"fullEval") ||
         isCasADi) {
        shared_ptr<void> value=eval.eval(e);
        if(value.get()) {//MFMF
          if(e->getFirstElementChild())
            e->removeChild(e->getFirstElementChild())->release();
          else if(E(e)->getFirstTextChild())
            e->removeChild(E(e)->getFirstTextChild())->release();
          DOMNode *node;
          DOMDocument *doc=e->getOwnerDocument();
          try {
            if(eval.getType(value)==Eval::SXFunctionType)
              node=eval.cast<DOMElement*>(value, doc);
            else
              node=doc->createTextNode(X()%eval.cast<string>(value));
          } MBXMLUTILS_RETHROW(e)
          e->appendChild(node);
        }
      }
    }
    
    // walk tree
    DOMElement *c=e->getFirstElementChild();
    while(c) {
      // pass param and the new parent XPath to preprocess
      preprocess(parser, eval, dependencies, c, param, parentXPath+"/"+thisXPath);
      if(c==NULL) break;
      c=c->getNextElementSibling();
    }
  } MBXMLUTILS_RETHROW(e);
}

}
