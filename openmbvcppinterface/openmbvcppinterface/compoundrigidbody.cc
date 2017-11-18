/*
    OpenMBV - Open Multi Body Viewer.
    Copyright (C) 2009 Markus Friedrich

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
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

CompoundRigidBody::CompoundRigidBody() : RigidBody(), expandStr("false") {
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
  e=E(element)->getFirstElementChildNamed(OPENMBV%"scaleFactor");
  e=e->getNextElementSibling();
  while (e) {
    std::shared_ptr<RigidBody> rb = ObjectFactory::create<RigidBody>(e);
    rb->initializeUsingXML(e);
    addRigidBody(rb);
    e=e->getNextElementSibling();
  }
}

}
