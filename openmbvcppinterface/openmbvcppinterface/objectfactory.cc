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

ObjectFactory& ObjectFactory::instance() {
  static ObjectFactory of;
  return of;
}

void ObjectFactory::deallocate(Object *obj) {
  obj->destroy();
}

void ObjectFactory::registerXMLName(const MBXMLUtils::FQN &name, allocateFkt alloc, deallocateFkt dealloc) {
  // check if name was already registred with the same &allocate<CreateType>: if yes return and do not add it twice
  std::pair<MapIt, MapIt> range=instance().registeredType.equal_range(name);
  for(MapIt it=range.first; it!=range.second; it++)
    if(it->second.first==alloc)
      return;
  // name is not registred with &allocate<CreateType>: register it
  instance().registeredType.insert(std::make_pair(name, std::make_pair(alloc, dealloc)));
}

}
