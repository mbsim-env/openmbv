/* Copyright (C) 2004-2010 OpenMBV Development Team
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

#include <config.h>
#include <openmbvcppinterface/objectfactory.h>
#include <openmbvcppinterface/object.h>

namespace OpenMBV {

std::string ObjectFactory::errorMsg;

ObjectFactory& ObjectFactory::instance() {
  static ObjectFactory of;
  return of;
}

void ObjectFactory::registerXMLName(const MBXMLUtils::FQN &name, allocateFkt alloc) {
  // check if name was already registred with the same &allocate<CreateType>: if yes return and do not add it twice
  std::pair<MapIt, MapIt> range=instance().registeredType.equal_range(name);
  for(auto it=range.first; it!=range.second; it++)
    if(it->second==alloc)
      return;
  // name is not registred with &allocate<CreateType>: register it
  instance().registeredType.insert(std::make_pair(name, alloc));
}

  void ObjectFactory::addErrorMsg(const std::string &msg) {
    errorMsg+=msg+"\n";
  }

  std::string ObjectFactory::getAndClearErrorMsg() {
    auto ret=errorMsg;
    errorMsg.clear();
    return ret;
  }

}
