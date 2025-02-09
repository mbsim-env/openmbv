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
#include <boost/functional/hash.hpp> //  boost::hash can hash a std::pair but std::hash cannot
#include <regex>
#ifdef _WIN32
  #ifndef WIN32_LEAN_AND_MEAN
  #  define WIN32_LEAN_AND_MEAN
  #endif
  #ifndef NOMINMAX
  #  define NOMINMAX
  #endif
  #include <windows.h>
  #undef __STRICT_ANSI__ // to define _controlfp which is not part of ANSI and hence not defined in mingw
  #include <cfloat>
  #define __STRICT_ANSI__
#else
  #include <cfenv>
#endif

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace MBXMLUtils {

  set<size_t> Deprecated::printedMessages;

  void Deprecated::message(const fmatvec::Atom *ele, const string &msg, const DOMElement *e) {
    // create the full deprecated message (including a trace)
    string msg2;
    if(e)
      msg2=DOMEvalException("", e).what();
    else
      // MISSING get a stacktrace here. e.g. using boost::backtrace if its available
      msg2="(no stack trace available)";
    auto &str = ele ? ele->msgStatic(fmatvec::Atom::Deprecated) : fmatvec::Atom::msgStatic(fmatvec::Atom::Deprecated);
    // create a hash of the message and ...
    boost::hash<pair<ostream*, string> > messageHash;
    if(printedMessages.insert(messageHash(make_pair(&str, msg+"\n"+msg2))).second) {
      // ... print the message if it is not already printed
      str<<"Deprecated feature called:"<<endl<<msg<<endl
         // skipws is not relevant for a ostream except by fmatvec::PrePostfixedStream which disabled the escaping
         // (that is why it is used here)
         <<flush<<skipws<<msg2<<flush<<noskipws
         <<endl;
    }
  }

  void setupMessageStreams(std::list<std::string> &args, bool forcePlainOutput) {
#ifdef _WIN32
    bool stdoutIsTTY=GetFileType(GetStdHandle(STD_OUTPUT_HANDLE))==FILE_TYPE_CHAR;
    bool stderrIsTTY=GetFileType(GetStdHandle(STD_ERROR_HANDLE))==FILE_TYPE_CHAR;
#else
    bool stdoutIsTTY=isatty(1)==1;
    bool stderrIsTTY=isatty(2)==1;
#endif
    // defaults for --stdout and --stderr
    if(find(args.begin(), args.end(), "--stdout")==args.end() &&
       find(args.begin(), args.end(), "--stderr")==args.end()) {
      if(  stdoutIsTTY && !forcePlainOutput ) { args.emplace_back("--stdout"); args.emplace_back(  "info~\x1b[KInfo: ~"); }
      if(  stderrIsTTY && !forcePlainOutput ) { args.emplace_back("--stderr"); args.emplace_back(  "warn~\x1b[KWarn: ~"); }
      if(  stderrIsTTY && !forcePlainOutput ) { args.emplace_back("--stderr"); args.emplace_back( "error~\x1b[K~"); }
      if(  stderrIsTTY && !forcePlainOutput ) { args.emplace_back("--stderr"); args.emplace_back(  "depr~\x1b[KDepr: ~"); }
      if(  stdoutIsTTY && !forcePlainOutput ) { args.emplace_back("--stdout"); args.emplace_back("status~\x1b[K~\r"); }

      if(!(stdoutIsTTY && !forcePlainOutput)) { args.emplace_back("--stdout"); args.emplace_back(  "info~Info: ~"); }
      if(!(stderrIsTTY && !forcePlainOutput)) { args.emplace_back("--stderr"); args.emplace_back(  "warn~Warn: ~"); }
      if(!(stderrIsTTY && !forcePlainOutput)) { args.emplace_back("--stderr"); args.emplace_back( "error~~"); }
      if(!(stderrIsTTY && !forcePlainOutput)) { args.emplace_back("--stderr"); args.emplace_back(  "depr~Depr: ~"); }
      if(!(stdoutIsTTY && !forcePlainOutput)) { args.emplace_back("--stdout"); args.emplace_back("status~~\n"); }
    }
  
    // disable all streams
    fmatvec::Atom::setCurrentMessageStream(fmatvec::Atom::Info      , std::make_shared<bool>(false));
    fmatvec::Atom::setCurrentMessageStream(fmatvec::Atom::Warn      , std::make_shared<bool>(false));
    fmatvec::Atom::setCurrentMessageStream(fmatvec::Atom::Debug     , std::make_shared<bool>(false));
    fmatvec::Atom::setCurrentMessageStream(fmatvec::Atom::Error     , std::make_shared<bool>(false));
    fmatvec::Atom::setCurrentMessageStream(fmatvec::Atom::Deprecated, std::make_shared<bool>(false));
    fmatvec::Atom::setCurrentMessageStream(fmatvec::Atom::Status    , std::make_shared<bool>(false));
  
    // handle --stdout and --stderr args
    list<string>::iterator it;
    while((it=find_if(args.begin(), args.end(), [](const string &x){ return x=="--stdout" || x=="--stderr"; }))!=args.end()) {
      ostream &ostr=*it=="--stdout"?cout:cerr;
      auto itn=next(it);
      if(itn==args.end())
        throw runtime_error("Invalid argument: "+*it+" "+*itn);
      fmatvec::Atom::MsgType msgType;
      if     (itn->substr(0, 5)=="info~"  ) msgType=fmatvec::Atom::Info;
      else if(itn->substr(0, 5)=="warn~"  ) msgType=fmatvec::Atom::Warn;
      else if(itn->substr(0, 6)=="debug~" ) msgType=fmatvec::Atom::Debug;
      else if(itn->substr(0, 6)=="error~" ) msgType=fmatvec::Atom::Error;
      else if(itn->substr(0, 5)=="depr~"  ) msgType=fmatvec::Atom::Deprecated;
      else if(itn->substr(0, 7)=="status~") msgType=fmatvec::Atom::Status;
      else throw runtime_error("Unknown message stream.");
      static std::regex re(".*~(.*)~(.*)", std::regex::extended);
      std::smatch m;
      auto value=*itn;
      bool html=false;
      if(value.substr(value.size()-5)=="~HTML") {
        html=true;
        value=value.substr(0, value.size()-5);
      }
      if(!std::regex_match(value, m, re))
        throw runtime_error("Invalid argument: "+*it+" "+*itn);
      fmatvec::Atom::setCurrentMessageStream(msgType, std::make_shared<bool>(true),
        std::make_shared<fmatvec::PrePostfixedStream>(m.str(1), m.str(2), ostr, html ? DOMEvalException::htmlEscaping : nullptr));
  
      args.erase(itn);
      args.erase(it);
    }
  }

  void handleFPE() {
    auto mbxmlutils_fpe=getenv("MBXMLUTILS_FPE");
    if(mbxmlutils_fpe && string(mbxmlutils_fpe)=="1") {
      #ifdef _WIN32
        _controlfp(~(_EM_ZERODIVIDE | _EM_INVALID | _EM_OVERFLOW), _MCW_EM);
      #else
        assert(feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW)!=-1);
      #endif
    }
  }

}
