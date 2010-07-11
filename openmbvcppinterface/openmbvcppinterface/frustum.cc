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

#include <openmbvcppinterface/frustum.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace OpenMBV;

Frustum::Frustum() : RigidBody(),
  baseRadius(1),
  topRadius(1),
  height(2),
  innerBaseRadius(0),
  innerTopRadius(0) {
}

TiXmlElement* Frustum::writeXMLFile(TiXmlNode *parent) {
  TiXmlElement *e=RigidBody::writeXMLFile(parent);
  addElementText(e, "baseRadius", baseRadius);
  addElementText(e, "topRadius", topRadius);
  addElementText(e, "height", height);
  addElementText(e, "innerBaseRadius", innerBaseRadius);
  addElementText(e, "innerTopRadius", innerTopRadius);
  return 0;
}

void Frustum::initializeUsingXML(TiXmlElement *element) {
  RigidBody::initializeUsingXML(element);
  TiXmlElement *e;
  e=element->FirstChildElement(OPENMBVNS"baseRadius");
  setBaseRadius(getDouble(e));
  e=element->FirstChildElement(OPENMBVNS"topRadius");
  setTopRadius(getDouble(e));
  e=element->FirstChildElement(OPENMBVNS"height");
  setHeight(getDouble(e));
  e=element->FirstChildElement(OPENMBVNS"innerBaseRadius");
  setInnerBaseRadius(getDouble(e));
  e=element->FirstChildElement(OPENMBVNS"innerTopRadius");
  setInnerTopRadius(getDouble(e));
}
