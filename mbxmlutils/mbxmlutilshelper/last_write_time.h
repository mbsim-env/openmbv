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
#include <sys/stat.h>
#ifdef _WIN32
#include <windows.h>
#endif

namespace boost::myfilesystem {

  inline boost::posix_time::ptime last_write_time(const std::string &p);
  inline void last_write_time(const std::string &p, const boost::posix_time::ptime &time);

  boost::posix_time::ptime last_write_time(const std::string &p) {
#ifndef _WIN32
    struct stat st;
    if(stat(p.c_str(), &st)!=0)
      throw std::runtime_error("system stat call failed: "+p);
    boost::posix_time::ptime time;
    time=boost::posix_time::from_time_t(st.st_mtime);
    time+=boost::posix_time::microsec(st.st_mtim.tv_nsec/1000);
    return time;
#else
    HANDLE f=CreateFile(p.c_str(), GENERIC_READ, FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if(f==INVALID_HANDLE_VALUE)
      throw std::runtime_error("CreateFile failed: "+p);
    FILETIME create, lastAccess, lastWrite;
    if(GetFileTime(f, &create, &lastAccess, &lastWrite)==0) {
      CloseHandle(f);
      throw std::runtime_error("GetFileTime failed: "+p);
    }
    CloseHandle(f);
    uint64_t microSecSince1601=((((uint64_t)(lastWrite.dwHighDateTime))<<32)+lastWrite.dwLowDateTime)/10;
    uint64_t hoursSince1601=microSecSince1601/1000000/60/60;
    return boost::posix_time::ptime(boost::gregorian::date(1601,boost::gregorian::Jan,1),
                                    boost::posix_time::hours(hoursSince1601)+
                                    boost::posix_time::microseconds(microSecSince1601-hoursSince1601*60*60*1000000));
#endif
  }
  void last_write_time(const std::string &p, const boost::posix_time::ptime &time) {
#ifndef _WIN32
    struct timeval times[2];
    boost::posix_time::time_period sinceEpoch(boost::posix_time::ptime(boost::gregorian::date(1970, boost::gregorian::Jan, 1)), time);
    times[0].tv_sec =sinceEpoch.length().total_seconds();
    times[0].tv_usec=sinceEpoch.length().total_microseconds()-1000000*times[0].tv_sec;
    times[1].tv_sec =times[0].tv_sec;
    times[1].tv_usec=times[0].tv_usec;
    if(utimes(p.c_str(), times)!=0)
      throw std::runtime_error("system utimes call failed: "+p);
#else
    HANDLE f=CreateFile(p.c_str(), FILE_WRITE_ATTRIBUTES, FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if(f==INVALID_HANDLE_VALUE)
      throw std::runtime_error("CreateFile failed: "+p);
    boost::posix_time::time_period since1601(boost::posix_time::ptime(boost::gregorian::date(1601, boost::gregorian::Jan, 1)), time);
    boost::posix_time::time_duration dt=since1601.length();
    uint64_t winTime=((uint64_t)(dt.hours()))*60*60*10000000;
    dt-=boost::posix_time::hours(dt.hours());
    winTime+=dt.total_microseconds()*10;
    FILETIME changeTime;
    changeTime.dwHighDateTime=(winTime>>32);
    changeTime.dwLowDateTime=(winTime & ((((uint64_t)1)<<32)-1));
    if(SetFileTime(f, NULL, &changeTime, &changeTime)==0) {
      CloseHandle(f);
      throw std::runtime_error("SetFileTime failed: "+p);
    }
    CloseHandle(f);
#endif
  }

}

#endif
