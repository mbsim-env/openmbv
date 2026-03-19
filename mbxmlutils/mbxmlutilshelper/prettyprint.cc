//#ifdef _WIN32
//#  define WIN32_LEAN_AND_MEAN
//#  include <windows.h>
//#endif
//#include <config.h>
//#include <clocale>
//#include <cassert>
//#include <cfenv>
//#include <iostream>
//#include <boost/filesystem/path.hpp>
//#include <boost/algorithm/string/predicate.hpp>
#include "dom.h"
#include "windows_signal_conversion.h"
#include "utils.h"

using namespace std;
using namespace boost::filesystem;
using namespace MBXMLUtils;
using namespace fmatvec;

int main(int argc, char *argv[]) {
#ifdef _WIN32
  SetConsoleCP(CP_UTF8);
  SetConsoleOutputCP(CP_UTF8);
#endif
  handleFPE();
  setlocale(LC_ALL, "C");
  convertWMCLOSEtoSIGTERM();

  vector<string> args;
  args.reserve(argc-1);
  for(int i=1; i<argc; ++i)
    args.emplace_back(argv[i]);

  if(args.size()==0) {
    cout<<"Usage: "<<argv[0]<<" <xml-file> [<xml-file> ...]"<<endl;
    cout<<"Parse and pretty-print all xml file given as arguments (the files are overwritten)"<<endl;
    return 0;
  }

  auto parser = DOMParser::create();

  int error=0;
  for(auto &arg : args) {
    path xml=arg;
    try {
      for(int pass : {1, 2}) {
        Atom::msgStatic(Atom::Info)<<xml.string()<<": pass "<<pass<<endl;
        auto startP = chrono::high_resolution_clock::now();
        auto doc = parser->parse(xml);
        auto endP = chrono::high_resolution_clock::now();
        Atom::msgStatic(Atom::Info)<<"- parsed in "<<chrono::duration<double>(endP-startP).count()*1000<<"ms"<<endl;
        auto startS = chrono::high_resolution_clock::now();
        DOMParser::serialize(doc.get(), xml, true);
        auto endS = chrono::high_resolution_clock::now();
        Atom::msgStatic(Atom::Info)<<"- serialized (pretty-printed) in "<<chrono::duration<double>(endS-startS).count()*1000<<"ms"<<endl;
      }
    }
    catch(const DOMEvalException &ex) {
      error++;
      // DOMEvalException is already passed thought escapeFunc -> skip escapeFunc (if enabled on the Atom streams) from duing another escaping
      Atom::msgStatic(Atom::Error)<<disableEscaping<<ex.what()<<enableEscaping<<endl;
    }
    catch(const std::exception &ex) {
      error++;
      Atom::msgStatic(Atom::Error)<<ex.what()<<endl;
    }
    catch(...) {
      error++;
      Atom::msgStatic(Atom::Error)<<"Unknown exception"<<endl;
    }
  }
  if(error>0)
    return 1;
  return 0;
}
