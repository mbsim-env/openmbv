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

#include <openmbvcppinterface/compoundrigidbody.h>
#include <openmbvcppinterface/objectfactory.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace OpenMBV;

CompoundRigidBody::CompoundRigidBody() : RigidBody() {
}

void CompoundRigidBody::writeXMLFile(std::ofstream& xmlFile, const std::string& indent) {
  xmlFile<<indent<<"<CompoundRigidBody name=\""<<name<<"\" enable=\""<<enableStr<<"\">"<<endl;
    RigidBody::writeXMLFile(xmlFile, indent+"  ");
    for(unsigned int i=0; i<rigidBody.size(); i++)
      rigidBody[i]->writeXMLFile(xmlFile, indent+"  ");
  xmlFile<<indent<<"</CompoundRigidBody>"<<endl;
}

void CompoundRigidBody::initializeUsingXML(TiXmlElement *element) {
  RigidBody::initializeUsingXML(element);
  TiXmlElement *e;
  e=element->FirstChildElement(OPENMBVNS"scaleFactor");
  e=e->NextSiblingElement();
  while (e) {
    RigidBody * rb = (RigidBody*)(ObjectFactory::createObject(e));
    rb->initializeUsingXML(e);
    addRigidBody(rb);
    e=e->NextSiblingElement();
  }
}
