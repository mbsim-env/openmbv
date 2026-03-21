#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/filesystem/fstream.hpp>
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif
#include "dom.h"
#include "windows_signal_conversion.h"
#include "utils.h"

using namespace std;
using namespace boost::filesystem;
using namespace MBXMLUtils;
using namespace fmatvec;

shared_ptr<DOMParser> parser;
bool quite = false;
bool skiperrors = false;
set<string> ext;

int processFile(const path& filename);

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
  for(int i=1; i<argc; ++i) {
    if(argv[i]=="-q"s || argv[i]=="--quite"s)
      quite = true;
    else if(argv[i]=="--skiperrors"s)
      skiperrors = true;
    else if(argv[i]=="--ext"s) {
      ++i;
      boost::split(ext, argv[i], boost::is_any_of(","));
    }
    else
      args.emplace_back(argv[i]);
  }

  if(args.size()==0) {
    cout<<"Usage: "<<argv[0]<<" [-q|--quite] [--skiperrors] [--ext <ext>,<ext>,...] <xml-file|dir> [<xml-file|dir> ...]"<<endl;
    cout<<"Parse and pretty-print all xml file given as arguments (the files are overwritten)"<<endl;
    cout<<"If the argument is a directory handle all files in this dir recursively"<<endl;
    cout<<"-q, --quite    Be quite, suppress informational messages"<<endl;
    cout<<"--skiperrors   Suppress error messages"<<endl;
    cout<<"--ext          For directories only handle the comma separated list of file extensions (include the .)"<<endl;
    return 0;
  }

  parser = DOMParser::create();

  int error = 0;
  for(auto &arg : args) {
    path filename=arg;
    if(!boost::filesystem::is_directory(filename))
      error += processFile(filename);
    else
      for(recursive_directory_iterator it(filename), end; it != end; ++it)
        if(is_regular_file(*it)) {
          if(ext.empty())
            error += processFile(it->path());
          else {
            for(auto &e : ext)
              if(boost::algorithm::ends_with(it->path().string(), e)) {
                error += processFile(it->path());
                break;
              }
          }
        }
  }
  if(error>0)
    return 1;
  return 0;
}

int processFile(const path& filename) {
  int error = 0;
  try {
    if(!quite)
      Atom::msgStatic(Atom::Info)<<filename.string()<<": processing"<<endl;

    auto startP1 = chrono::high_resolution_clock::now();
    auto doc1 = parser->parse(filename, nullptr, false);
    auto endP1 = chrono::high_resolution_clock::now();
    if(!quite)
      Atom::msgStatic(Atom::Info)<<"- parsed file in "<<chrono::duration<double>(endP1-startP1).count()*1000<<"ms"<<endl;

    auto startS1 = chrono::high_resolution_clock::now();
    string dataOut1;
    DOMParser::serializeToString(doc1.get(), dataOut1, true);
    auto endS1 = chrono::high_resolution_clock::now();
    if(!quite)
      Atom::msgStatic(Atom::Info)<<"- serialized (pretty-printed) in "<<chrono::duration<double>(endS1-startS1).count()*1000<<"ms"<<endl;

    auto startP2 = chrono::high_resolution_clock::now();
    istringstream is(dataOut1);
    auto doc2 = parser->parse(is, nullptr, false, filename);
    auto endP2 = chrono::high_resolution_clock::now();
    if(!quite)
      Atom::msgStatic(Atom::Info)<<"- reparsed in "<<chrono::duration<double>(endP2-startP2).count()*1000<<"ms"<<endl;

    auto startS2 = chrono::high_resolution_clock::now();
    string dataOut;
    DOMParser::serializeToString(doc2.get(), dataOut, true);
    auto endS2 = chrono::high_resolution_clock::now();
    if(!quite)
      Atom::msgStatic(Atom::Info)<<"- reserialized (pretty-printed) in "<<chrono::duration<double>(endS2-startS2).count()*1000<<"ms"<<endl;

    auto startR = chrono::high_resolution_clock::now();
    string dataIn;
    {
      boost::filesystem::ifstream file(filename);
      dataIn = string((istreambuf_iterator<char>(file)), istreambuf_iterator<char>());
    }
    auto endR = chrono::high_resolution_clock::now();
    if(!quite)
      Atom::msgStatic(Atom::Info)<<"- read file in "<<chrono::duration<double>(endR-startR).count()*1000<<"ms"<<endl;

    auto startH1 = chrono::high_resolution_clock::now();
    size_t hashIn = std::hash<string>{}(dataIn);
    auto endH1 = chrono::high_resolution_clock::now();
    if(!quite)
      Atom::msgStatic(Atom::Info)<<"- calculated input hash in "<<chrono::duration<double>(endH1-startH1).count()*1000<<"ms"<<endl;

    auto startH2 = chrono::high_resolution_clock::now();
    size_t hashOut = std::hash<string>{}(dataOut);
    auto endH2 = chrono::high_resolution_clock::now();
    if(!quite)
      Atom::msgStatic(Atom::Info)<<"- calculated output hash in "<<chrono::duration<double>(endH2-startH2).count()*1000<<"ms"<<endl;

    if(hashIn==hashOut) {
      if(!quite)
        Atom::msgStatic(Atom::Info)<<"- hash not changed, no write needed"<<endl;
    }
    else {
      auto startW = chrono::high_resolution_clock::now();
      {
        boost::filesystem::ofstream file(filename);
        file.write(dataOut.data(), dataOut.size());
      }
      auto endW = chrono::high_resolution_clock::now();
      if(!quite)
        Atom::msgStatic(Atom::Info)<<"- wrote file in "<<chrono::duration<double>(endW-startW).count()*1000<<"ms"<<endl;
      else
        Atom::msgStatic(Atom::Info)<<filename<<": pretty printed"<<endl;
    }
  }
  catch(const DOMEvalException &ex) {
    error++;
    // DOMEvalException is already passed thought escapeFunc -> skip escapeFunc (if enabled on the Atom streams) from duing another escaping
    if(!skiperrors)
      Atom::msgStatic(Atom::Error)<<disableEscaping<<ex.what()<<enableEscaping<<endl;
  }
  catch(const std::exception &ex) {
    error++;
    if(!skiperrors)
      Atom::msgStatic(Atom::Error)<<ex.what()<<endl;
  }
  catch(...) {
    error++;
    if(!skiperrors)
      Atom::msgStatic(Atom::Error)<<"Unknown exception"<<endl;
  }
  return error;
}
