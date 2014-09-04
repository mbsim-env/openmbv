#include <config.h>
#include "last_write_time.h"
#include <sys/stat.h>
#ifdef _WIN32
#include <windows.h>
#endif

void dummyfunclsdkjflksdjf() {} // required to generate a object file: seems to be a compiler bug

using namespace std;

namespace boost {
  namespace myfilesystem {
    boost::posix_time::ptime last_write_time(const string &p) {
#ifndef _WIN32
      struct stat st;
      if(stat(p.c_str(), &st)!=0)
        throw runtime_error("system stat call failed: "+p);
      boost::posix_time::ptime time;
      time=boost::posix_time::from_time_t(st.st_mtime);
      time+=boost::posix_time::microsec(st.st_mtim.tv_nsec/1000);
      return time;
#else
      HANDLE f=CreateFile(p.c_str(), GENERIC_READ, FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
      if(f==INVALID_HANDLE_VALUE)
        throw runtime_error("CreateFile failed: "+p);
      FILETIME create, lastAccess, lastWrite;
      if(GetFileTime(f, &create, &lastAccess, &lastWrite)==0) {
        CloseHandle(f);
        throw runtime_error("GetFileTime failed: "+p);
      }
      CloseHandle(f);
      uint64_t microSecSince1601=((((uint64_t)(lastWrite.dwHighDateTime))<<32)+lastWrite.dwLowDateTime)/10;
      uint64_t hoursSince1601=microSecSince1601/1000000/60/60;
      return boost::posix_time::ptime(boost::gregorian::date(1601,boost::gregorian::Jan,1),
                                      boost::posix_time::hours(hoursSince1601)+
                                      boost::posix_time::microseconds(microSecSince1601-hoursSince1601*60*60*1000000));
#endif
    }
    void last_write_time(const string &p, const boost::posix_time::ptime &time) {
#ifndef _WIN32
      struct timeval times[2];
      boost::posix_time::time_period sinceEpoch(boost::posix_time::ptime(boost::gregorian::date(1970, boost::gregorian::Jan, 1)), time);
      times[0].tv_sec =sinceEpoch.length().total_seconds();
      times[0].tv_usec=sinceEpoch.length().total_microseconds()-1000000*times[0].tv_sec;
      times[1].tv_sec =times[0].tv_sec;
      times[1].tv_usec=times[0].tv_usec;
      if(utimes(p.c_str(), times)!=0)
        throw runtime_error("system utimes call failed: "+p);
#else
      HANDLE f=CreateFile(p.c_str(), FILE_WRITE_ATTRIBUTES, FILE_SHARE_DELETE | FILE_SHARE_READ | FILE_SHARE_WRITE, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
      if(f==INVALID_HANDLE_VALUE)
        throw runtime_error("CreateFile failed: "+p);
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
        throw runtime_error("SetFileTime failed: "+p);
      }
      CloseHandle(f);
#endif
    }
  }
}
