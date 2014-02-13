#ifndef _MBXMLUTILS_LAST_WRITE_TIME_H_
#define _MBXMLUTILS_LAST_WRITE_TIME_H_

/* This is a varaint of the boost::filesystem::last_write_time functions.
 * It only differs in the argument/return value being here a boost::posix_time::ptime instead of a time_t.
 * This enables file timestamps on microsecond level.
 * We use type string for argument p (instead of boost::filesystem::path) here to avoid a dependency to boost::filesystem here. */
#include <boost/date_time/posix_time/posix_time.hpp>
#include <string>

namespace boost {
  namespace myfilesystem {

    boost::posix_time::ptime last_write_time(const std::string &p);
    void last_write_time(const std::string &p, const boost::posix_time::ptime &time);

  }
}

#endif
