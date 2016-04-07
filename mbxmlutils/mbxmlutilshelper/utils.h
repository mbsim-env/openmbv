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

#ifndef _MBXMLUTILSHELPER_UTILS_H_
#define _MBXMLUTILSHELPER_UTILS_H_

#include <fmatvec/atom.h>
#include <string>
#include <set>
#include <vector>
#include <xercesc/dom/DOMElement.hpp>

namespace MBXMLUtils {

class Deprecated : virtual public fmatvec::Atom {
  public:
    /*! register a deprecated feature with name message.
     * If e is NULL a stack trace is printed if available if e it not NULL MBXMLUtils::DOMEvalException is printed. */
    static void registerMessage(const std::string &message, const xercesc::DOMElement *e=NULL);
//  private:
//    ~Deprecated();
//    static Deprecated& getInstance();
//    std::set<std::vector<std::string> > allMessages;
};

}

#endif
