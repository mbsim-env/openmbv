#ifndef MBXMLUTILS_SHAREDLIBNAME
#  error "MBXMLUTILS_SHAREDLIBNAME must be defined before including this implementation file."
#endif

#ifdef _WIN32
#  include <windows.h>
#else
#  ifndef _GNU_SOURCE
#    define _GNU_SOURCE // dladdr requires _GNU_SOURCE to be defined
#  endif
#  include <dlfcn.h>
#endif

namespace {

#ifdef _WIN32
extern "C" void *__ImageBase;
#else
char buffer[2048];
std::string pathAtLoadTime=getcwd(buffer, sizeof(buffer));
#endif

}

namespace MBXMLUtils {

std::string BOOST_PP_CAT(get, BOOST_PP_CAT(MBXMLUTILS_SHAREDLIBNAME, SharedLibPath))() {
  static std::string ret;
  if(!ret.empty())
    return ret;

  // get the shared library file path containing this function
#ifdef _WIN32
  char moduleName[2048];
  GetModuleFileName(reinterpret_cast<HMODULE>(&__ImageBase), moduleName, sizeof(moduleName));
  ret=moduleName;
#else
  Dl_info info;
  dladdr(reinterpret_cast<void*>(&BOOST_PP_CAT(get, BOOST_PP_CAT(MBXMLUTILS_SHAREDLIBNAME, SharedLibPath))), &info);
  // convert to absolute path and return
  std::string name(info.dli_fname);
  ret=name[0]=='/'?name:BOOST_PP_CAT(pathAtLoadTime, MBXMLUTILS_SHAREDLIBNAME)+"/"+name;
#endif
  return ret;
}

}

#undef MBXMLUTILS_SHAREDLIBNAME
