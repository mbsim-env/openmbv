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

#include <openmbvcppinterface/ivbody.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace OpenMBV;

IvBody::IvBody() : RigidBody(), creaseAngle(-1), boundaryEdges(false) {
}

TiXmlElement* IvBody::writeXMLFile(TiXmlNode *parent) {
  TiXmlElement *e=RigidBody::writeXMLFile(parent);
  addElementText(e, "ivFileName", "\""+ivFileName+"\"");
  addElementText(e, "creaseAngle", creaseAngle, -1);
  addElementText(e, "boundaryEdges", boundaryEdges, false);
  return 0;
}

void IvBody::initializeUsingXML(TiXmlElement *element) {
  RigidBody::initializeUsingXML(element);
  TiXmlElement *e;
  e=element->FirstChildElement(OPENMBVNS"ivFileName");
  setIvFileName(string(e->GetText()).substr(1,string(e->GetText()).length()-2));
  e=element->FirstChildElement(OPENMBVNS"creaseAngle");
  if(e) setCreaseEdges(getDouble(e));
  e=element->FirstChildElement(OPENMBVNS"boundaryEdges");
  if(e) setBoundaryEdges((e->GetText()==string("true") || e->GetText()==string("1"))?true:false);
}
