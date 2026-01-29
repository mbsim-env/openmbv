#ifndef _MBSIMXML_SET_CURRENT_PATH_H_
#define _MBSIMXML_SET_CURRENT_PATH_H_

#include <boost/filesystem.hpp>
#include <mbxmlutilshelper/utils.h>

class SetCurrentPath {
  public:
    SetCurrentPath(const boost::filesystem::path& newCurrentPath) {
      orgCurrentPath=MBXMLUtils::current_path();
      if(!newCurrentPath.empty())
        MBXMLUtils::current_path(newCurrentPath);
    }
    boost::filesystem::path adaptPath(const boost::filesystem::path &p) const {
      if(p.is_absolute())
        return p;
      return boost::filesystem::relative(orgCurrentPath/p);
    }
    bool existsInOrg(const boost::filesystem::path &p) const {
      if(p.is_absolute())
        return boost::filesystem::exists(p);
      return boost::filesystem::exists(orgCurrentPath/p);
    }
  private:
    boost::filesystem::path orgCurrentPath;
};

#endif
