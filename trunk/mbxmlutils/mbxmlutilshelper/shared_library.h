#ifndef _MBXMLUTILS_SHAREDLIBRARY_H_
#define _MBXMLUTILS_SHAREDLIBRARY_H_

#include <boost/filesystem.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

namespace MBXMLUtils {

class SharedLibrary {
  public:
    SharedLibrary(const boost::filesystem::path &file_);
    SharedLibrary(const SharedLibrary& src);
    ~SharedLibrary();
    void* getAddress(const std::string &symbolName);
    const boost::filesystem::path file;
    const boost::posix_time::ptime writeTime;
    bool operator<(const SharedLibrary& b) const { return file<b.file; }
  private:
    std::string getLastError();
    void init();
#ifndef _WIN32
    void* handle;
#else
    HMODULE handle;
#endif
};

}

#endif
