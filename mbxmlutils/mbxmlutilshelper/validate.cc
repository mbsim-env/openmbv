#include <config.h>
#include <cassert>
#include <cfenv>
#include <iostream>
#include <boost/filesystem/path.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include "dom.h"

using namespace std;
using namespace boost::filesystem;
using namespace MBXMLUtils;

int main(int argc, char *argv[]) {
#ifndef _WIN32
  assert(feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW)!=-1);
#endif

  vector<string> args;
  args.reserve(argc-1);
  for(int i=1; i<argc; ++i)
    args.emplace_back(argv[i]);

  if(args.size()==0) {
    cout<<"Usage: "<<argv[0]<<" --xmlCatalog <catalog.xml> <xml-file> [<xml-file> ...]"<<endl;
    return 0;
  }

  path xmlCatalog;
  auto it=find(args.begin(), args.end(), "--xmlCatalog");
  if(it!=args.end()) {
    auto itn=it;
    itn++;
    xmlCatalog=*itn;
    args.erase(itn);
    args.erase(it);
  }

  shared_ptr<DOMParser> parser;
  try {
    parser=DOMParser::create(xmlCatalog);
  }
  catch(const std::exception &ex) {
    fmatvec::Atom::msgStatic(fmatvec::Atom::Error)<<"Exception while loading schemas:"<<endl
         <<ex.what()<<endl;
    return 1;
  }

  int error=0;
  for(auto &arg : args) {
    path xml=arg;
    try {
      parser->parse(xml);
      fmatvec::Atom::msgStatic(fmatvec::Atom::Info)<<xml<<" validates."<<endl;
    }
    catch(const std::exception &ex) {
      error++;
      fmatvec::Atom::msgStatic(fmatvec::Atom::Error)<<ex.what()<<endl;
    }
  }
  if(error>0)
    return 1;
  return 0;
}
