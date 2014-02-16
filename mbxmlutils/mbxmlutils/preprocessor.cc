#include <mbxmlutilshelper/dom.h>
#include <list>
#include <stdexcept>
#include <iostream>
#include <octeval.h>
#include <mbxmlutilshelper/getinstallpath.h>
#include <boost/locale.hpp>
#include <boost/bind.hpp>
#include <xercesc/dom/DOMDocument.hpp>
#include <xercesc/dom/DOMNamedNodeMap.hpp>
#include <xercesc/dom/DOMAttr.hpp>
#include <xercesc/dom/DOMException.hpp>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;
using namespace boost;
using namespace boost::filesystem;

path SCHEMADIR;

void preprocess(shared_ptr<DOMParser> parser, OctEval &octEval, vector<path> &dependencies, DOMElement *&e) {
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
        string subst=OctEval::cast<string>(ret);
        if(OctEval::getType(ret)==OctEval::StringType)
          subst=subst.substr(1, subst.length()-2);
        file=absolute(subst, E(e)->getOriginalFilename().parent_path());
        dependencies.push_back(file);
      }
  
      // evaluate count using parameters
      long count=1;
      if(E(e)->hasAttribute("count"))
        count=OctEval::cast<long>(octEval.eval(E(e)->getAttributeNode("count"), e));
  
      // couter name
      string counterName="MBXMLUtilsDummyCounterName";
      if(E(e)->hasAttribute("counterName")) {
        counterName=OctEval::cast<string>(octEval.eval(E(e)->getAttributeNode("counterName"), e));
        counterName=counterName.substr(1, counterName.length()-2);
      }
  
      shared_ptr<DOMElement> enew;
      // validate/load if file is given
      if(!file.empty()) {
        cout<<"Read and validate "<<file<<endl;
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
        cout<<"Generate local octave parameters for "<<(file.empty()?"<inline element>":file)<<endl;
        if(inlineParamEle) // inline parameter
          octEval.addParamSet(inlineParamEle);
        else { // parameter from parameterHref attribute
          octave_value ret=octEval.eval(E(e)->getAttributeNode("parameterHref"), e);
          string subst=OctEval::cast<string>(ret);
          if(OctEval::getType(ret)==OctEval::StringType)
            subst=subst.substr(1, subst.length()-2);
          path paramFile=absolute(subst, E(e)->getOriginalFilename().parent_path());
          // add local parameter file to dependencies
          dependencies.push_back(paramFile);
          // validate and local parameter file
          cout<<"Read and validate local parameter file "<<paramFile<<endl;
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
          onlyif=(OctEval::cast<long>(octEval.eval(E(e)->getAttributeNode("onlyif"), e))==1);
        if(onlyif) {
          cout<<"Embed "<<(file.empty()?"<inline element>":file)<<" ("<<i<<"/"<<count<<")"<<endl;
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
          cout<<"Skip embeding "<<(file.empty()?"<inline element>":file)<<" ("<<i<<"/"<<count<<"); onlyif attribute is false"<<endl;
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
        string s=OctEval::cast<string>(value);
        if(OctEval::getType(value)==OctEval::StringType)
          s=s.substr(1, s.length()-2);
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
          if(OctEval::getType(value)==OctEval::SXFunctionType)
            node=OctEval::cast<DOMElement*>(value, doc);
          else
            node=doc->createTextNode(X()%OctEval::cast<string>(value));
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
  }
  catch(const DOMEvalException &ex) {
    throw ex;
  }
  catch(const DOMEvalExceptionList &ex) {
    throw ex;
  }
  catch(const std::exception &ex) {
    throw DOMEvalException(ex.what(), e);
  }
}

int main(int argc, char *argv[]) {
  InitXerces initXerces;
  try {

    // use UTF8 globally (expecially in path)
    std::locale::global(locale::generator().generate("UTF8"));
    path::imbue(std::locale());

    // convert argv to list
    list<string> arg;
    for(int i=1; i<argc; i++)
      arg.push_back(argv[i]);

    // help message
    if(arg.size()<3) {
      cout<<"Usage:"<<endl
          <<"mbxmlutilspp [--dependencies <dep-file-name>]"<<endl
          <<"              <param-file> [dir/]<main-file> <namespace-location-of-main-file>"<<endl
          <<"             [<param-file> [dir/]<main-file> <namespace-location-of-main-file>]"<<endl
          <<"             ..."<<endl
          <<""<<endl
          <<"  --dependencies    Write a newline separated list of dependent files including"<<endl
          <<"                    <param-file> and <main-file> to <dep-file-name>"<<endl
          <<""<<endl
          <<"  The output file is named '.pp.<main-file>'."<<endl
          <<"  Use 'none' if not <param-file> is avaliabel."<<endl
          <<""<<endl
          <<"Copyright (C) 2009 Markus Friedrich <friedrich.at.gc@googlemail.com>"<<endl
          <<"This is free software; see the source for copying conditions. There is NO"<<endl
          <<"warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE."<<endl
          <<""<<endl
          <<"Licensed under the GNU Lesser General Public License (LGPL)"<<endl;
      return 0;
    }

    // a global oct evaluator just to prevent multiple init/deinit calles
    OctEval globalOctEval;

    // the XML DOM parser
    shared_ptr<DOMParser> parser=DOMParser::create(true);

    list<string>::iterator i, i2;

    // dependency file
    vector<path> dependencies;
    path depFileName;
    if((i=std::find(arg.begin(), arg.end(), "--dependencies"))!=arg.end()) {
      i2=i; i2++;
      depFileName=(*i2);
      arg.erase(i); arg.erase(i2);
    }

    SCHEMADIR=getInstallPath()/"share"/"mbxmlutils"/"schema";

    // load global schema grammar
    cout<<"Load XML grammar for parameters."<<endl;
    parser->loadGrammar(SCHEMADIR/"http___openmbv_berlios_de_MBXMLUtils"/"physicalvariable.xsd");

    // loop over all files
    while(arg.size()>0) {
      // initialize the parameter stack (clear ALL caches)
      OctEval octEval(&dependencies);

      path paramxml(*arg.begin()); arg.erase(arg.begin());
      path mainxml(*arg.begin()); arg.erase(arg.begin());
      path nslocation(*arg.begin()); arg.erase(arg.begin());

      // validate parameter file and get DOM
      shared_ptr<xercesc::DOMDocument> paramxmldoc;
      if(paramxml!="none") {
        cout<<"Read and validate "<<paramxml<<endl;
        paramxmldoc=parser->parse(paramxml);
        dependencies.push_back(paramxml);
      }

      // generate octave parameter string
      if(paramxmldoc.get()) {
        cout<<"Generate octave parameter set from "<<paramxml<<endl;
        octEval.addParamSet(paramxmldoc->getDocumentElement());
      }

      // load grammar
      cout<<"Load XML grammar for main file (cached if loaded multiple times)."<<endl;
      parser->loadGrammar(nslocation);

      // validate main file and get DOM
      cout<<"Read and validate "<<mainxml<<endl;
      shared_ptr<xercesc::DOMDocument> mainxmldoc=parser->parse(mainxml);
      dependencies.push_back(mainxml);

      // embed/validate/toOctave/unit/eval files
      DOMElement *mainxmlele=mainxmldoc->getDocumentElement();
      preprocess(parser, octEval, dependencies, mainxmlele);

      // save result file
      path mainxmlpp=".pp."+mainxml.filename().string();
      cout<<"Save preprocessed file "<<mainxml<<" as "<<mainxmlpp<<endl;
      DOMParser::serialize(mainxmldoc.get(), mainxmlpp, false);
      cout<<"Validate preprocessed file"<<endl;
      parser->parse(mainxmlpp); // = D(mainxmldoc)->validate() (serialization is already done)
    }

    // output dependencies?
    if(!depFileName.empty()) {
      ofstream dependenciesFile(depFileName.string().c_str());
      for(vector<path>::iterator it=dependencies.begin(); it!=dependencies.end(); it++)
        dependenciesFile<<it->string()<<endl;
    }
  }
  catch(const std::exception &ex) {
    cerr<<ex.what()<<endl;
    return 1;
  }
  catch(const DOMException &ex) {
    cerr<<"DOM exception: "<<X()%ex.getMessage()<<endl;
    return 1;
  }
  catch(...) {
    cerr<<"Unknown exception."<<endl;
    return 1;
  }
  return 0;
}
