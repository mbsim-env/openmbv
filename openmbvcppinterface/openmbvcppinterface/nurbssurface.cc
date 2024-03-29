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
#include <openmbvcppinterface/nurbssurface.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(NurbsSurface, OPENMBV%"NurbsSurface")

DOMElement* NurbsSurface::writeXMLFile(DOMNode *parent) {
  DOMElement *e=RigidBody::writeXMLFile(parent);
  E(e)->addElementText(OPENMBV%"controlPoints", cp);
  E(e)->addElementText(OPENMBV%"numberOfUControlPoints", numU);
  E(e)->addElementText(OPENMBV%"numberOfVControlPoints", numV);
  E(e)->addElementText(OPENMBV%"uKnotVector", uKnot);
  E(e)->addElementText(OPENMBV%"vKnotVector", vKnot);
  return nullptr;
}

void NurbsSurface::initializeUsingXML(DOMElement *element) {
  RigidBody::initializeUsingXML(element);
  DOMElement *e=E(element)->getFirstElementChildNamed(OPENMBV%"controlPoints");
  setControlPoints(E(e)->getText<vector<vector<double>>>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"numberOfUControlPoints");
  setNumberOfUControlPoints(E(e)->getText<int>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"numberOfVControlPoints");
  setNumberOfVControlPoints(E(e)->getText<int>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"uKnotVector");
  setUKnotVector(E(e)->getText<vector<double>>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"vKnotVector");
  setVKnotVector(E(e)->getText<vector<double>>());
}

}
