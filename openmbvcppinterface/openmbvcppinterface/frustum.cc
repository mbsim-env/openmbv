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

void Frustum::writeXMLFile(std::ofstream& xmlFile, const std::string& indent) {
  xmlFile<<indent<<"<Frustum name=\""<<name<<"\">"<<endl;
    RigidBody::writeXMLFile(xmlFile, indent+"  ");
    xmlFile<<indent<<"  <baseRadius>"<<baseRadius<<"</baseRadius>"<<endl;
    xmlFile<<indent<<"  <topRadius>"<<topRadius<<"</topRadius>"<<endl;
    xmlFile<<indent<<"  <height>"<<height<<"</height>"<<endl;
    xmlFile<<indent<<"  <innerBaseRadius>"<<innerBaseRadius<<"</innerBaseRadius>"<<endl;
    xmlFile<<indent<<"  <innerTopRadius>"<<innerTopRadius<<"</innerTopRadius>"<<endl;
  xmlFile<<indent<<"</Frustum>"<<endl;
}

void Frustum::initializeUsingXML(TiXmlElement *element) {
  RigidBody::initializeUsingXML(element);
  TiXmlElement *e;
  e=element->FirstChildElement(OPENMBVNS"baseRadius");
  setBaseRadius(atof(e->GetText()));
  e=element->FirstChildElement(OPENMBVNS"topRadius");
  setTopRadius(atof(e->GetText()));
  e=element->FirstChildElement(OPENMBVNS"height");
  setHeight(atof(e->GetText()));
  e=element->FirstChildElement(OPENMBVNS"innerBaseRadius");
  setInnerBaseRadius(atof(e->GetText()));
  e=element->FirstChildElement(OPENMBVNS"innerTopRadius");
  setInnerTopRadius(atof(e->GetText()));
}
