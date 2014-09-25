#include <config.h>
#include "mbxmlutils/preprocess.h"
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMAttr.hpp>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;
using namespace boost;
using namespace boost::filesystem;

namespace MBXMLUtils {

void Preprocess::preprocess(shared_ptr<DOMParser> parser, OctEval &octEval, vector<path> &dependencies, DOMElement *&e) {
  try {
    if(E(e)->getTagName()==PV%"Embed") {
      NewParamLevel newParamLevel(octEval);
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
        octave_value ret=octEval.eval(E(e)->getAttributeNode("href"), e);
        string subst;
        try {
          subst=OctEval::cast<string>(ret);
          if(OctEval::getType(ret)==OctEval::StringType)
            subst=subst.substr(1, subst.length()-2);
        } MBXMLUTILS_RETHROW(e)
        file=E(e)->convertPath(subst);
        dependencies.push_back(file);
      }
    
      // evaluate count using parameters
      long count=1;
      if(E(e)->hasAttribute("count"))
        try { count=OctEval::cast<long>(octEval.eval(E(e)->getAttributeNode("count"), e)); } MBXMLUTILS_RETHROW(e)
    
      // couter name
      string counterName="MBXMLUtilsDummyCounterName";
      if(E(e)->hasAttribute("counterName")) {
        try { counterName=OctEval::cast<string>(octEval.eval(E(e)->getAttributeNode("counterName"), e)); } MBXMLUTILS_RETHROW(e)
        counterName=counterName.substr(1, counterName.length()-2);
      }
    
      shared_ptr<DOMElement> enew;
      // validate/load if file is given
      if(!file.empty()) {
        octEval.msg(Info)<<"Read and validate "<<file<<endl;
        shared_ptr<DOMDocument> newdoc=parser->parse(file);
        E(newdoc->getDocumentElement())->workaroundDefaultAttributesOnImportNode();// workaround
        enew.reset(static_cast<DOMElement*>(e->getOwnerDocument()->importNode(newdoc->getDocumentElement(), true)),
          bind(&DOMElement::release, _1));
      }
      else { // or take the child element (inlineEmbedEle)
        enew.reset(static_cast<DOMElement*>(e->removeChild(inlineEmbedEle)),
          bind(&DOMElement::release, _1));
      }
    
      // include a processing instruction with the line number of the original element
      E(enew)->setOriginalElementLineNumber(E(e)->getLineNumber());
    
      // generate local paramter for embed
      // check that not both the parameterHref attribute and the child element pv:Parameter exists (This is not checked by the schema)
      DOMElement *inlineParamEle=e->getFirstElementChild();
      if(inlineParamEle && E(inlineParamEle)->getTagName()!=PV%"Parameter")
        inlineParamEle=NULL;
      if(inlineParamEle && E(e)->hasAttribute("parameterHref"))
        throw DOMEvalException("Only the parameterHref attribute OR the child element pv:Parameter is allowed in Embed!", e);
      if(inlineParamEle || E(e)->hasAttribute("parameterHref")) {
        octEval.msg(Info)<<"Generate local octave parameters for "<<(file.empty()?"<inline element>":file)<<endl;
        if(inlineParamEle) // inline parameter
          octEval.addParamSet(inlineParamEle);
        else { // parameter from parameterHref attribute
          octave_value ret=octEval.eval(E(e)->getAttributeNode("parameterHref"), e);
          string subst;
          try {
            subst=OctEval::cast<string>(ret);
            if(OctEval::getType(ret)==OctEval::StringType)
              subst=subst.substr(1, subst.length()-2);
          } MBXMLUTILS_RETHROW(e)
          path paramFile=E(e)->convertPath(subst);
          // add local parameter file to dependencies
          dependencies.push_back(paramFile);
          // validate and local parameter file
          octEval.msg(Info)<<"Read and validate local parameter file "<<paramFile<<endl;
          shared_ptr<DOMDocument> localparamxmldoc=parser->parse(paramFile);
          // generate local parameters
          octEval.addParamSet(localparamxmldoc->getDocumentElement());
        }
      }
    
      // delete embed element and insert count time the new element
      for(long i=1; i<=count; i++) {
        octEval.addParam(counterName, i);

        // embed only if 'onlyif' attribute is true
        bool onlyif=true;
        if(E(e)->hasAttribute("onlyif"))
          try { onlyif=(OctEval::cast<long>(octEval.eval(E(e)->getAttributeNode("onlyif"), e))==1); } MBXMLUTILS_RETHROW(e)
        if(onlyif) {
          octEval.msg(Info)<<"Embed "<<(file.empty()?"<inline element>":file)<<" ("<<i<<"/"<<count<<")"<<endl;
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
          preprocess(parser, octEval, dependencies, e);
        }
        else
          octEval.msg(Info)<<"Skip embeding "<<(file.empty()?"<inline element>":file)<<" ("<<i<<"/"<<count<<"); onlyif attribute is false"<<endl;
      }
      return;
    }
    else if(E(e)->getTagName()==CasADi::CASADI%"SXFunction")
      return; // skip processing of SXFunction elements
    else {
      bool isCasADi=false;

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
        if(!A(a)->isDerivedFrom(PV%"fullOctEval") && !A(a)->isDerivedFrom(PV%"partialOctEval"))
          continue;
        octave_value value=octEval.eval(a, e);
        string s;
        try {
          s=OctEval::cast<string>(value);
          if(OctEval::getType(value)==OctEval::StringType)
            s=s.substr(1, s.length()-2);
        } MBXMLUTILS_RETHROW(e)
        a->setValue(X()%s);
      }

      // evaluate element if it must be evaluated
      if(E(e)->isDerivedFrom(PV%"scalar") ||
         E(e)->isDerivedFrom(PV%"vector") ||
         E(e)->isDerivedFrom(PV%"matrix") ||
         E(e)->isDerivedFrom(PV%"fullOctEval") ||
         isCasADi) {
        octave_value value=octEval.eval(e);
        if(!value.is_empty()) {
          if(e->getFirstElementChild())
            e->removeChild(e->getFirstElementChild())->release();
          else if(E(e)->getFirstTextChild())
            e->removeChild(E(e)->getFirstTextChild())->release();
          DOMNode *node;
          DOMDocument *doc=e->getOwnerDocument();
          try {
            if(OctEval::getType(value)==OctEval::SXFunctionType)
              node=OctEval::cast<DOMElement*>(value, doc);
            else
              node=doc->createTextNode(X()%OctEval::cast<string>(value));
          } MBXMLUTILS_RETHROW(e)
          e->appendChild(node);
        }
      }
    }
    
    // walk tree
    DOMElement *c=e->getFirstElementChild();
    while(c) {
      preprocess(parser, octEval, dependencies, c);
      if(c==NULL) break;
      c=c->getNextElementSibling();
    }
  } MBXMLUTILS_RETHROW(e);
}

}
