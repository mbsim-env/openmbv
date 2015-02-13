#ifndef MBXMLUTILS_SHAREDLIBNAME
#  error "MBXMLUTILS_SHAREDLIBNAME must be defined before including this header file."
#endif

#include <boost/filesystem.hpp>

namespace MBXMLUtils {

//! Retrun the absolute path of the so/dll file where this function is defined
boost::filesystem::path BOOST_PP_CAT(get, BOOST_PP_CAT(MBXMLUTILS_SHAREDLIBNAME, SharedLibPath))();

}

#undef MBXMLUTILS_SHAREDLIBNAME
