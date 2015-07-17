/*
 * Author: Markus Friedrich
 *
 * This file is free and unencumbered software released into the public domain.
 * 
 * Anyone is free to copy, modify, publish, use, compile, sell, or
 * distribute this software, either in source code form or as a compiled
 * binary, for any purpose, commercial or non-commercial, and by any
 * means.
 * 
 * In jurisdictions that recognize copyright laws, the author or authors
 * of this software dedicate any and all copyright interest in the
 * software to the public domain. We make this dedication for the benefit
 * of the public at large and to the detriment of our heirs and
 * successors. We intend this dedication to be an overt act of
 * relinquishment in perpetuity of all present and future rights to this
 * software under copyright law.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 * 
 * For more information, please refer to <http://unlicense.org/>
 */

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
