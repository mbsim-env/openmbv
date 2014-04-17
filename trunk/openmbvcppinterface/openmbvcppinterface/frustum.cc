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
#include <openmbvcppinterface/frustum.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(Frustum, OPENMBV%"Frustum")

Frustum::Frustum() : RigidBody(),
  baseRadius(1),
  topRadius(1),
  height(2),
  innerBaseRadius(0),
  innerTopRadius(0) {
}

DOMElement* Frustum::writeXMLFile(DOMNode *parent) {
  DOMElement *e=RigidBody::writeXMLFile(parent);
  addElementText(e, OPENMBV%"baseRadius", baseRadius);
  addElementText(e, OPENMBV%"topRadius", topRadius);
  addElementText(e, OPENMBV%"height", height);
  addElementText(e, OPENMBV%"innerBaseRadius", innerBaseRadius);
  addElementText(e, OPENMBV%"innerTopRadius", innerTopRadius);
  return 0;
}

void Frustum::initializeUsingXML(DOMElement *element) {
  RigidBody::initializeUsingXML(element);
  DOMElement *e;
  e=E(element)->getFirstElementChildNamed(OPENMBV%"baseRadius");
  setBaseRadius(getDouble(e));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"topRadius");
  setTopRadius(getDouble(e));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"height");
  setHeight(getDouble(e));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"innerBaseRadius");
  setInnerBaseRadius(getDouble(e));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"innerTopRadius");
  setInnerTopRadius(getDouble(e));
}

}
