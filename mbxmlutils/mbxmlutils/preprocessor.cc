#include "config.h"
#include <cassert>
#include <cfenv>
#include <boost/regex.hpp>
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
#ifndef _WIN32
  assert(feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW)!=-1);
#endif

  try {

    //TODO not working on Windows
    //TODO // use UTF8 globally (expecially in path)
    //TODO std::locale::global(locale::generator().generate("UTF8"));
    //TODO path::imbue(std::locale());

    // convert argv to list
    list<string> args;
    for(int i=1; i<argc; i++)
      args.emplace_back(argv[i]);

    // help message
    if(args.size()<2) {
      cout<<"Usage:"<<endl
          <<"mbxmlutilspp [--dependencies <dep-file-name>]"<<endl
          <<"             [--stdout <msg> [--stdout <msg> ...]] [--stderr <msg> [--stderr <msg> ...]]"<<endl
          <<"             <schema-file.xsd> [<schema-file.xsd> ...]"<<endl
          <<"             <main-file> [<main-file> ...]"<<endl
          <<"             ..."<<endl
          <<""<<endl
          <<"  --dependencies    Write a newline separated list of dependent files including"<<endl
          <<"                    <param-file> and <main-file> to <dep-file-name>"<<endl
          <<"  --stdout <msg>    Print on stdout messages of type <msg>."<<endl
          <<"                    <msg> may be info~<pre>~<post>, warn~<pre>~<post>, debug~<pre>~<post>"<<endl
          <<"                    error~<pre>~<post>~ or depr~<pre>~<post>~."<<endl
          <<"                    Each message is prefixed/postfixed with <pre>/<post>."<<endl
          <<"                    --stdout may be specified multiple times."<<endl
          <<"                    If --stdout and --stderr is not specified --stdout 'info~Info: ~'"<<endl
          <<"                    --stderr 'warn~Warn: ~' --stderr 'error~~' --stderr 'depr~Depr:~'"<<endl
          <<"                    --stderr 'status~~\\r' is used."<<endl
          <<"  --stderr <msg>    Analog to --stdout but prints to stderr."<<endl
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

    // defaults for --stdout and --stderr
    if(find(args.begin(), args.end(), "--stdout")==args.end() &&
       find(args.begin(), args.end(), "--stderr")==args.end()) {
      args.push_back("--stdout"); args.push_back("info~Info: ~");
      args.push_back("--stderr"); args.push_back("warn~Warn: ~");
      args.push_back("--stderr"); args.push_back("error~~");
      args.push_back("--stderr"); args.push_back("depr~Depr: ~");
      args.push_back("--stdout"); args.push_back("status~~\r");
    }

    // disable all streams
    fmatvec::Atom::setCurrentMessageStream(fmatvec::Atom::Info      , std::make_shared<bool>(false));
    fmatvec::Atom::setCurrentMessageStream(fmatvec::Atom::Warn      , std::make_shared<bool>(false));
    fmatvec::Atom::setCurrentMessageStream(fmatvec::Atom::Debug     , std::make_shared<bool>(false));
    fmatvec::Atom::setCurrentMessageStream(fmatvec::Atom::Error     , std::make_shared<bool>(false));
    fmatvec::Atom::setCurrentMessageStream(fmatvec::Atom::Deprecated, std::make_shared<bool>(false));
    fmatvec::Atom::setCurrentMessageStream(fmatvec::Atom::Status    , std::make_shared<bool>(false));

    // handle --stdout and --stderr args
    list<string>::iterator it;
    while((it=find_if(args.begin(), args.end(), [](const string &x){ return x=="--stdout" || x=="--stderr"; }))!=args.end()) {
      ostream &ostr=*it=="--stdout"?cout:cerr;
      auto itn=next(it);
      if(itn==args.end()) {
        cerr<<"Invalid argument"<<endl;
        return 1;
      }
      fmatvec::Atom::MsgType msgType;
      if     (itn->substr(0, 5)=="info~"  ) msgType=fmatvec::Atom::Info;
      else if(itn->substr(0, 5)=="warn~"  ) msgType=fmatvec::Atom::Warn;
      else if(itn->substr(0, 6)=="debug~" ) msgType=fmatvec::Atom::Debug;
      else if(itn->substr(0, 6)=="error~" ) msgType=fmatvec::Atom::Error;
      else if(itn->substr(0, 5)=="depr~"  ) msgType=fmatvec::Atom::Deprecated;
      else if(itn->substr(0, 7)=="status~") msgType=fmatvec::Atom::Status;
      else throw runtime_error("Unknown message stream.");
      static boost::regex re(".*~(.*)~(.*)", boost::regex::extended);
      boost::smatch m;
      if(!boost::regex_match(*itn, m, re)) {
        cerr<<"Invalid argument"<<endl;
        return 1;
      }
      fmatvec::Atom::setCurrentMessageStream(msgType, std::make_shared<bool>(true),
        std::make_shared<fmatvec::PrePostfixedStream>(m.str(1), m.str(2), ostr));

      args.erase(itn);
      args.erase(it);
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
    fmatvec::Atom::msgStatic(fmatvec::Atom::Info)<<"Create validating XML parser."<<endl;
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
      fmatvec::Atom::msgStatic(fmatvec::Atom::Info)<<"Read and validate "<<mainxml<<endl;
      shared_ptr<xercesc::DOMDocument> mainxmldoc=parser->parse(mainxml, &dependencies, false);
      dependencies.push_back(mainxml);
      DOMElement *mainxmlele=mainxmldoc->getDocumentElement();

      // create a clean evaluator (get the evaluator name first form the dom)
      string evalName="octave"; // default evaluator
      DOMElement *evaluator;
      if(E(mainxmlele)->getTagName()==PV%"Embed") {
        // if the root element IS A Embed than the <evaluator> element is the first child of the
        // first (none pv:Parameter) child of the root element
        auto r=mainxmlele->getFirstElementChild();
        if(E(r)->getTagName()==PV%"Parameter")
          r=r->getNextElementSibling();
        evaluator=E(r)->getFirstElementChildNamed(PV%"evaluator");
      }
      else
        // if the root element IS NOT A Embed than the <evaluator> element is the first child root element
        evaluator=E(mainxmlele)->getFirstElementChildNamed(PV%"evaluator");
      if(evaluator)
        evalName=X()%E(evaluator)->getFirstTextChild()->getData();
      shared_ptr<Eval> eval=Eval::createEvaluator(evalName, &dependencies);

      // embed/validate/unit/eval files
      Preprocess::preprocess(parser, eval, dependencies, mainxmlele);

      // adapt the evaluator in the dom (reset evaluator because it may change if the root element is a Embed)
      evaluator=E(mainxmlele)->getFirstElementChildNamed(PV%"evaluator");
      if(evaluator)
        E(evaluator)->getFirstTextChild()->setData(X()%"xmlflat");
      else {
        evaluator=D(mainxmldoc)->createElement(PV%"evaluator");
        evaluator->appendChild(mainxmldoc->createTextNode(X()%"xmlflat"));
        mainxmlele->insertBefore(evaluator, mainxmlele->getFirstChild());
      }

      // save result file
      path mainxmlpp=".pp."+mainxml.filename().string();
      fmatvec::Atom::msgStatic(fmatvec::Atom::Info)<<"Save preprocessed file "<<mainxml<<" as "<<mainxmlpp<<endl;
      E(mainxmlele)->setOriginalFilename();
      DOMParser::serialize(mainxmldoc.get(), mainxmlpp, false);
      fmatvec::Atom::msgStatic(fmatvec::Atom::Info)<<"Validate preprocessed file"<<endl;
      parser->parse(mainxmlpp, nullptr, false); // = D(mainxmldoc)->validate() (serialization is already done)
    }

    // output dependencies?
    if(!depFileName.empty()) {
      std::ofstream dependenciesFile(depFileName.string().c_str());
      for(auto & dependencie : dependencies)
        dependenciesFile<<dependencie.string()<<endl;
    }
  }
  catch(const std::exception &ex) {
    fmatvec::Atom::msgStatic(fmatvec::Atom::Error)<<ex.what()<<endl;
    return 1;
  }
  catch(...) {
    fmatvec::Atom::msgStatic(fmatvec::Atom::Error)<<"Unknown exception."<<endl;
    return 1;
  }
  return 0;
}
