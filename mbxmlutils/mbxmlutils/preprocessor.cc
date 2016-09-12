#include "config.h"
#include "mbxmlutils/preprocess.h"
#include <mbxmlutilshelper/getinstallpath.h>
#include <mbxmlutils/eval.h>
#include <xercesc/dom/DOMDocument.hpp>
#include <boost/algorithm/string/predicate.hpp>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;
using namespace boost::filesystem;

path SCHEMADIR;

int main(int argc, char *argv[]) {
  try {

    //TODO not working on Windows
    //TODO // use UTF8 globally (expecially in path)
    //TODO std::locale::global(locale::generator().generate("UTF8"));
    //TODO path::imbue(std::locale());

    // convert argv to list
    list<string> args;
    for(int i=1; i<argc; i++)
      args.push_back(argv[i]);

    // help message
    if(args.size()<2) {
      cout<<"Usage:"<<endl
          <<"mbxmlutilspp [--dependencies <dep-file-name>]"<<endl
          <<"             <schema-file.xsd> [<schema-file.xsd> ...]"<<endl
          <<"             <main-file> [<main-file> ...]"<<endl
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

    list<string>::iterator i, i2;

    // dependency file
    vector<path> dependencies;
    path depFileName;
    if((i=std::find(args.begin(), args.end(), "--dependencies"))!=args.end()) {
      i2=i; i2++;
      depFileName=(*i2);
      args.erase(i); args.erase(i2);
    }

    SCHEMADIR=getInstallPath()/"share"/"mbxmlutils"/"schema";

    // the XML DOM parser
    cout<<"Create validating XML parser."<<endl;
    set<path> schemas={SCHEMADIR/"http___www_mbsim-env_de_MBXMLUtils"/"physicalvariable.xsd"};
    for(auto &arg: args) {
      if(boost::algorithm::ends_with(arg, ".xsd"))
        schemas.insert(arg);
    }
    shared_ptr<DOMParser> parser=DOMParser::create(schemas);

    // loop over all main files
    for(auto &arg: args) {
      if(boost::algorithm::ends_with(arg, ".xsd"))
        continue;

      path mainxml(arg);

      // validate main file and get DOM
      cout<<"Read and validate "<<mainxml<<endl;
      shared_ptr<xercesc::DOMDocument> mainxmldoc=parser->parse(mainxml, &dependencies);
      dependencies.push_back(mainxml);
      DOMElement *mainxmlele=mainxmldoc->getDocumentElement();

      // create a clean evaluator (get the evaluator name first form the dom)
      string evalName="octave"; // default evaluator
      DOMElement *evaluator=E(mainxmlele)->getFirstElementChildNamed(PV%"evaluator");
      if(evaluator)
        evalName=X()%E(evaluator)->getFirstTextChild()->getData();
      shared_ptr<Eval> eval=Eval::createEvaluator(evalName, &dependencies);

      // embed/validate/unit/eval files
      Preprocess::preprocess(parser, eval, dependencies, mainxmlele);

      // adapt the evaluator in the dom
      if(evaluator)
        E(evaluator)->getFirstTextChild()->setData(X()%"xmlflat");
      else {
        evaluator=D(mainxmldoc)->createElement(PV%"evaluator");
        evaluator->appendChild(mainxmldoc->createTextNode(X()%"xmlflat"));
        mainxmlele->insertBefore(evaluator, mainxmlele->getFirstChild());
      }

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
    cerr<<"Exception:"<<endl
        <<ex.what()<<endl;
    return 1;
  }
  catch(...) {
    cerr<<"Unknown exception."<<endl;
    return 1;
  }
  return 0;
}
