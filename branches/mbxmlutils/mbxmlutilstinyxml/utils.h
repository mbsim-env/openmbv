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

#ifndef _MBXMLUTILSTINYXML_UTILS_H_
#define _MBXMLUTILSTINYXML_UTILS_H_

#include <string>
#include <set>
#include <vector>
#include "mbxmlutilstinyxml/tinyxml.h"

namespace MBXMLUtils {

class Deprecated {
  public:
    /*! register a deprecated feature with name message.
     * If e is NULL a stack trace is printed if available if e it not NULL MBXMLUtils::TiXml_location is printed. */
    static void registerMessage(const std::string &message, const MBXMLUtils::TiXmlElement *e=NULL);
  private:
    static void printAllMessages();
    static std::set<std::vector<std::string> > allMessages;
    static bool atExitRegistred;
};

std::string demangleSymbolName(std::string name);

}

#endif
