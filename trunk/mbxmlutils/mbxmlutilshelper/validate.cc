#include <config.h>
#include <iostream>
#include <boost/filesystem/path.hpp>
#include "dom.h"

using namespace std;
using namespace boost;
using namespace boost::filesystem;
using namespace MBXMLUtils;

int main(int argc, char *argv[]) {
  path schema=argv[1];
  shared_ptr<DOMParser> parser=DOMParser::create(true);
  try {
    parser->loadGrammar(schema);
  }
  catch(const std::exception &ex) {
    cerr<<"Exception:"<<endl
         <<ex.what()<<endl;
    return 1;
  }

  int error=0;
  for(int i=2; i<argc; ++i) {
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
