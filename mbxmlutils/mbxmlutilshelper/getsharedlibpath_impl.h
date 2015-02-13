#ifndef MBXMLUTILS_SHAREDLIBNAME
#  error "MBXMLUTILS_SHAREDLIBNAME must be defined before including this implementation file."
#endif

#include <boost/filesystem.hpp>

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
boost::filesystem::path pathAtLoadTime=boost::filesystem::current_path();
#endif

}

namespace MBXMLUtils {

boost::filesystem::path BOOST_PP_CAT(get, BOOST_PP_CAT(MBXMLUTILS_SHAREDLIBNAME, SharedLibPath))() {
  // get the shared library file path containing this function
#ifdef _WIN32
  wchar_t moduleName[2048];
  GetModuleFileNameW(reinterpret_cast<HMODULE>(&__ImageBase), moduleName, sizeof(moduleName));
  return moduleName;
#else
  Dl_info info;
  dladdr(reinterpret_cast<void*>(&BOOST_PP_CAT(get, BOOST_PP_CAT(MBXMLUTILS_SHAREDLIBNAME, SharedLibPath))), &info);
  return boost::filesystem::absolute(info.dli_fname, pathAtLoadTime);
#endif
}

}

#undef MBXMLUTILS_SHAREDLIBNAME
