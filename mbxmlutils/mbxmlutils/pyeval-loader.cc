#include <string>
#include <mbxmlutilshelper/thislinelocation.h>
#include <mbxmlutilshelper/shared_library.h>
#include <mbxmlutils/pycppwrapper_mainlib.h>
#include <boost/filesystem.hpp>

using namespace std;
using namespace MBXMLUtils;

// This is just a helper library which is needed to load the python evaluator on Linux.
// It just calls the ctor code of Init which
// on Windows:
// Just loads the "real" python eval library mbxmlutils-eval-main-python.dll
// on Linux:
// Checks if python is already loaded to the global symbol namespace and if not it load python to the global symbol namespace
// Then the "real" python eval library libmbxmlutils-eval-main-python.so is loaded

namespace {

  ThisLineLocation loc;

  class Init {
    public:
      Init();
  };

  Init::Init() {
    auto installPath = boost::filesystem::path(loc()).parent_path().parent_path();
#ifdef _WIN32
    string libname="bin/libmbxmlutils-eval-python-runtime.dll";
#else
    auto [PYMAINLIB, PYMAINLIBFILE, PYTHONLOADED] = PythonCpp::getPythonMainLib(installPath.string());
    if(!PYTHONLOADED)
      SharedLibrary::load((boost::filesystem::path(PYMAINLIB)/PYMAINLIBFILE).string(), true);
    string libname="lib/libmbxmlutils-eval-python-runtime.so";
#endif
    SharedLibrary::load((installPath/libname).string());
  }

  Init init;

}
