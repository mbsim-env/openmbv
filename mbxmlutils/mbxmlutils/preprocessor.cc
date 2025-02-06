#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  undef __STRICT_ANSI__ // to define _controlfp which is not part of ANSI and hence not defined in mingw
#  include <cfloat>
#  define __STRICT_ANSI__
#endif
#include "config.h"
#include <clocale>
#include <cassert>
#include <fstream>
#include <cfenv>
#include <regex>
#include "mbxmlutils/preprocess.h"
#include <mbxmlutilshelper/utils.h>
#include <mbxmlutils/eval.h>
#include <xercesc/dom/DOMDocument.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <mbxmlutilshelper/windows_signal_conversion.h>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;
using namespace boost::filesystem;

int main(int argc, char *argv[]) {
#ifdef _WIN32
  SetConsoleCP(CP_UTF8);
  SetConsoleOutputCP(CP_UTF8);
  _controlfp(~(_EM_ZERODIVIDE | _EM_INVALID | _EM_OVERFLOW), _MCW_EM);
#else
  assert(feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW)!=-1);
#endif
  setlocale(LC_ALL, "C");
  convertWMCLOSEtoSIGTERM();

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
          <<"             --xmlCatalog <catalog.xml>"<<endl
          <<"             -o <out-file> <main-file>"<<endl
          <<"             ..."<<endl
          <<""<<endl
          <<"  --dependencies    Write a newline separated list of dependent files including"<<endl
          <<"                    <main-file> to <dep-file-name>"<<endl
          <<"  --stdout <msg>    Print on stdout messages of type <msg>."<<endl
          <<"                    <msg> may be 'info~<pre>~<post>', 'warn~<pre>~<post>', 'debug~<pre>~<post>'"<<endl
          <<"                    'error~<pre>~<post>' or 'depr~<pre>~<post>'."<<endl
          <<"                    Each message is prefixed/postfixed with <pre>/<post>."<<endl
          <<"                    If '~HTML' is appended than HTML escaping takes place."<<endl
          <<"                    --stdout may be specified multiple times."<<endl
          <<"  --stderr <msg>    Analog to --stdout but prints to stderr."<<endl
          <<""<<endl
          <<"  The output is written to <out-file>, if -o is given, or to stdout."<<endl
          <<""<<endl
          <<"Copyright (C) 2009 Markus Friedrich <friedrich.at.gc@googlemail.com>"<<endl
          <<"This is free software; see the source for copying conditions. There is NO"<<endl
          <<"warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE."<<endl
          <<""<<endl
          <<"Licensed under the GNU Lesser General Public License (LGPL)"<<endl;
      return 0;
    }

    // handle --stdout and --stderr args
    setupMessageStreams(args);

    list<string>::iterator i, i2;

    // dependency file
    vector<path> dependencies;
    path depFileName;
    if((i=std::find(args.begin(), args.end(), "--dependencies"))!=args.end()) {
      i2=i; i2++;
      depFileName=(*i2);
      args.erase(i); args.erase(i2);
    }

    auto it=find(args.begin(), args.end(), "--xmlCatalog");
    it++;
    path xmlCatalog=*it;
    path mainXML(args.back());

    Preprocess preprocess(mainXML, xmlCatalog, !depFileName.empty());
    auto mainXMLDoc = preprocess.processAndGetDocument();
    if(!depFileName.empty())
      dependencies = preprocess.getDependencies();

    // save result file
    path mainxmlpp;
    if((i=std::find(args.begin(), args.end(), "-o"))!=args.end())
      mainxmlpp=*(++i);
    else
      throw runtime_error("No -o argument given.");
    fmatvec::Atom::msgStatic(fmatvec::Atom::Info)<<"Save preprocessed file "<<mainXML<<" as "<<mainxmlpp<<endl;
    DOMElement *mainxmlele=mainXMLDoc->getDocumentElement();
    E(mainxmlele)->setOriginalFilename();
    DOMParser::serialize(mainXMLDoc.get(), mainxmlpp);

    // output dependencies?
    if(!depFileName.empty()) {
      boost::filesystem::ofstream dependenciesFile(depFileName);
      for(auto & dependencie : dependencies)
        dependenciesFile<<dependencie.string()<<endl;
    }
  }
  catch(const DOMEvalException &ex) {
    // DOMEvalException is already passed thought escapeFunc -> skip escapeFunc (if enabled on the fmatvec::Atom streams) from duing another escaping
    fmatvec::Atom::msgStatic(fmatvec::Atom::Error)<<flush<<skipws<<ex.what()<<flush<<noskipws<<endl;
    return 1;
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
