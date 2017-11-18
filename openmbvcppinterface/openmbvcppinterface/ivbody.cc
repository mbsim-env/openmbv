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
#include <openmbvcppinterface/ivbody.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(IvBody, OPENMBV%"IvBody")

IvBody::IvBody() : RigidBody() {
}

DOMElement* IvBody::writeXMLFile(DOMNode *parent) {
  DOMElement *e=RigidBody::writeXMLFile(parent);
  E(e)->addElementText(OPENMBV%"ivFileName", "'"+ivFileName+"'");
  E(e)->addElementText(OPENMBV%"creaseEdges", creaseAngle);
  E(e)->addElementText(OPENMBV%"boundaryEdges", boundaryEdges);
  return nullptr;
}

void IvBody::initializeUsingXML(DOMElement *element) {
  RigidBody::initializeUsingXML(element);
  DOMElement *e;
  e=E(element)->getFirstElementChildNamed(OPENMBV%"ivFileName");
  string str = X()%E(e)->getFirstTextChild()->getData();
  setIvFileName(E(e)->convertPath(str.substr(1,str.length()-2)).string());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"creaseEdges");
  if(e) setCreaseEdges(E(e)->getText<double>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"boundaryEdges");
  if(e) {
    string str = X()%E(e)->getFirstTextChild()->getData();
    setBoundaryEdges((str=="true" || str=="1")?true:false);
  }
}

}
