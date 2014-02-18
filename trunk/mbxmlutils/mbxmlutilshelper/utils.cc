/* Copyright (C) 2004-2009 OpenMBV Development Team
 *
 * This library is free software; you can redistribute it and/or 
 * modify it under the terms of the GNU Lesser General Public 
 * License as published by the Free Software Foundation; either 
 * version 2.1 of the License, or (at your option) any later version. 
 *  
 * This library is distributed in the hope that it will be useful, 
 * but WITHOUT ANY WARRANTY; without even the implied warranty of 
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
 * Lesser General Public License for more details. 
 *  
 * You should have received a copy of the GNU Lesser General Public 
 * License along with this library; if not, write to the Free Software 
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
 *
 * Contact: friedrich.at.gc@googlemail.com
 */

#include "config.h"
#include <cstdlib>
#include "utils.h"
#include "dom.h"
#if defined HAVE_LIBUNWIND_H && defined HAVE_LIBUNWIND
#  include <libunwind.h>
#endif
#if defined HAVE_CXXABI_H
#  include <cxxabi.h>
#endif

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace MBXMLUtils {

  set<vector<string> > Deprecated::allMessages;
  bool Deprecated::atExitRegistred=false;

  void Deprecated::registerMessage(const std::string &message, const DOMElement *e) {
    if(!atExitRegistred) {
      atexit(&Deprecated::printAllMessages);
      atExitRegistred=true;
    }

    vector<string> stack;
    stack.push_back(message);
    if(e)
      stack.push_back(DOMEvalException("", e).what());
    else {
#if defined HAVE_LIBUNWIND_H && defined HAVE_LIBUNWIND
      try {
        unw_context_t context;
        if(unw_getcontext(&context)<0) throw 1;
        unw_cursor_t cp;
        if(unw_init_local(&cp, &context)<0) throw 1;
        if(unw_step(&cp)<=0) throw 1;
        unw_word_t offp;
        char name[102400];
        int nr=0;
        do {
          if(unw_get_proc_name(&cp, name, 102400, &offp)<0) break;
          stack.push_back((nr==0?"at ":"by ")+demangleSymbolName(name));
          nr++;
        }
        while(unw_step(&cp)>0 && string(name)!="main");
      }
      catch(...) {
        stack.push_back("(no stack trace available)");
      }
#else
      stack.push_back("(no stack trace available)");
#endif
    }
    allMessages.insert(stack);
  }

  void Deprecated::printAllMessages() {
    cerr<<endl;
    cerr<<"WARNING: "<<allMessages.size()<<" deprecated features were called:"<<endl;
    set<vector<string> >::const_iterator it;
    int nr=0;
    for(it=allMessages.begin(); it!=allMessages.end(); it++) {
      nr++;
      cerr<<"* "<<"("<<nr<<"/"<<allMessages.size()<<") "<<(*it)[0]<<endl;
      vector<string>::const_iterator it2=it->begin();
      it2++;
      for(; it2!=it->end(); it2++)
        cerr<<"  "<<*it2<<endl;
    }
  }

  std::string demangleSymbolName(std::string name) {
#if defined HAVE_CXXABI_H
    std::string ret=name;
    int status=1;
    char *demangledName=NULL;
    demangledName=abi::__cxa_demangle(name.c_str(), NULL, NULL, &status);
    if(status==0)
      ret=demangledName;
    free(demangledName);
    return ret;
#else
    return name;
#endif
  }

}
