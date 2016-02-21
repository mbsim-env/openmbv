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

  Deprecated& Deprecated::getInstance() {
    static Deprecated instance;
    return instance;
  }

  Deprecated::~Deprecated() {
    msg(Warn)<<"\n";
    msg(Warn)<<allMessages.size()<<" deprecated features were called:\n";
    set<vector<string> >::const_iterator it;
    int nr=0;
    for(it=allMessages.begin(); it!=allMessages.end(); it++) {
      nr++;
      msg(Warn)<<"* "<<"("<<nr<<"/"<<allMessages.size()<<") "<<(*it)[0]<<"\n";
      vector<string>::const_iterator it2=it->begin();
      it2++;
      for(; it2!=it->end(); it2++)
        msg(Warn)<<"  "<<*it2<<"\n";
    }
    msg(Warn)<<endl;
  }

  void Deprecated::registerMessage(const std::string &message, const DOMElement *e) {
    vector<string> stack;
    stack.push_back(message);
    if(e)
      stack.push_back(DOMEvalException("", e).what());
    else {
#if defined HAVE_LIBUNWIND_H && defined HAVE_LIBUNWIND
      try {
        unw_context_t context;
        if(unw_getcontext(&context)<0) throw runtime_error("Internal error: Unable to get unwind context");
        unw_cursor_t cp;
        if(unw_init_local(&cp, &context)<0) throw runtime_error("Internal error: Unable to init local unwind");
        if(unw_step(&cp)<=0) throw runtime_error("Internal error: Unable to step unwind");
        unw_word_t offp;
        char name[102400];
        int nr=0;
        do {
          if(unw_get_proc_name(&cp, name, 102400, &offp)<0) break;
          stack.push_back((nr==0?"at ":"by ")+boost::core::demangle(name));
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
    getInstance().allMessages.insert(stack);
  }

}
