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
#include <boost/functional/hash.hpp>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace MBXMLUtils {

  set<size_t> Deprecated::printedMessages;

  void Deprecated::message(ostream &str, string msg, const DOMElement *e) {
    // create the full deprecated message (including a trace)
    if(e)
      msg+=string("\n")+DOMEvalException("", e).what();
    else
      // MISSING get a stacktrace here. e.g. using boost::backtrace if its available
      msg+="\n(no stack trace available)";
    // create a hash of the message and ...
    boost::hash<pair<ostream*, string> > messageHash;
    if(printedMessages.insert(messageHash(make_pair(&str, msg))).second)
      // ... print the message if it is not already printed
      str<<endl<<"Deprecated feature called:"<<endl<<msg<<endl<<endl;
  }

}
