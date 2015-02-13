#include <config.h>
#include "getinstallpath.h"

#define MBXMLUTILS_SHAREDLIBNAME MBXMLUtilsHelper
#include "getsharedlibpath_impl.h"

using namespace boost::filesystem;

namespace MBXMLUtils {

path getInstallPath() {
  return path(getMBXMLUtilsHelperSharedLibPath()).parent_path().parent_path();
}

} // end namespace MBXMLUtils
