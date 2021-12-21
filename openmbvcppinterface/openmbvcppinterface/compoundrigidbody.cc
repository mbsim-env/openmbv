/*
    OpenMBV - Open Multi Body Viewer.
    Copyright (C) 2009 Markus Friedrich

  This library is free software; you can redistribute it and/or 
  modify it under the terms of the GNU Lesser General Public 
  License as published by the Free Software Foundation; either 
  version 2.1 of the License, or (at your option) any later version. 
   
  This library is distributed in the hope that it will be useful, 
  but WITHOUT ANY WARRANTY; without even the implied warranty of 
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
  Lesser General Public License for more details. 
   
  You should have received a copy of the GNU Lesser General Public 
  License along with this library; if not, write to the Free Software 
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
*/

#include "config.h"
#include <openmbvcppinterface/compoundrigidbody.h>
#include <openmbvcppinterface/objectfactory.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(CompoundRigidBody, OPENMBV%"CompoundRigidBody")

CompoundRigidBody::CompoundRigidBody() :  expandStr("false") {
}

CompoundRigidBody::~CompoundRigidBody() = default;

DOMElement* CompoundRigidBody::writeXMLFile(DOMNode *parent) {
  DOMElement *e=RigidBody::writeXMLFile(parent);
  E(e)->setAttribute("expand", expandStr);
  for(auto & i : rigidBody)
    i->writeXMLFile(e);
  return nullptr;
}

void CompoundRigidBody::initializeUsingXML(DOMElement *element) {
  RigidBody::initializeUsingXML(element);
  if(E(element)->hasAttribute("expand") && 
     (E(element)->getAttribute("expand")=="true" || E(element)->getAttribute("expand")=="1"))
    setExpand(true);
  DOMElement *e;

  // search last element of RigidBody
  // TODO/MISSING: initializeUsingXML(...) should return the first unparsed DOMElement!
  // Then we can avoid such ugly hacks to detect the element here which needs to know all base class definitions.
         e=E(element)->getFirstElementChildNamed(OPENMBV%"scaleFactor");
  if(!e) e=E(element)->getFirstElementChildNamed(OPENMBV%"initialRotation");
  if(!e) e=E(element)->getFirstElementChildNamed(OPENMBV%"initialTranslation");
  if(!e) e=E(element)->getFirstElementChildNamed(OPENMBV%"transparency");
  if(!e) e=E(element)->getFirstElementChildNamed(OPENMBV%"diffuseColor");
  if(!e) e=E(element)->getFirstElementChildNamed(OPENMBV%"maximalColorValue");
  if(!e) e=E(element)->getFirstElementChildNamed(OPENMBV%"minimalColorValue");
  if(e)
    e=e->getNextElementSibling();
  else
    e=element->getFirstElementChild();

  while (e) {
    std::shared_ptr<RigidBody> rb = ObjectFactory::create<RigidBody>(e);
    rb->initializeUsingXML(e);
    addRigidBody(rb);
    e=e->getNextElementSibling();
  }
}

}
