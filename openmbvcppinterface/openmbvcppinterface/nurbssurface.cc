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
