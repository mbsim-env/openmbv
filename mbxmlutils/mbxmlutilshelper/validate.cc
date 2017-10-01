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

  set<path> schemas;
  for(int i=1; i<argc; ++i)
    if(boost::algorithm::ends_with(argv[i], ".xsd"))
      schemas.insert(argv[i]);

  shared_ptr<DOMParser> parser;
  try {
    parser=DOMParser::create(schemas);
  }
  catch(const std::exception &ex) {
    cerr<<"Exception while loading schemas:"<<endl
         <<ex.what()<<endl;
    return 1;
  }

  int error=0;
  for(int i=1; i<argc; ++i) {
    if(boost::algorithm::ends_with(argv[i], ".xsd"))
      continue;

    path xml=argv[i];
    try {
      parser->parse(xml);
      cerr<<xml<<" validates."<<endl;
    }
    catch(const std::exception &ex) {
      error++;
      cerr<<"Exception:"<<endl
          <<ex.what()<<endl;
    }
  }
  if(error>0)
    return 1;
  return 0;
}
