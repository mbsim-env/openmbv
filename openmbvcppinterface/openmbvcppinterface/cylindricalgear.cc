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
#include <openmbvcppinterface/cylindricalgear.h>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(CylindricalGear, OPENMBV%"CylindricalGear")

DOMElement *CylindricalGear::writeXMLFile(DOMNode *parent) {
  DOMElement *e=RigidBody::writeXMLFile(parent);
  E(e)->addElementText(OPENMBV%"numberOfTeeth", N);
  E(e)->addElementText(OPENMBV%"width", w);
  E(e)->addElementText(OPENMBV%"helixAngle", be);
  E(e)->addElementText(OPENMBV%"module", m);
  E(e)->addElementText(OPENMBV%"pressureAngle", al);
  E(e)->addElementText(OPENMBV%"backlash", b);
  E(e)->addElementText(OPENMBV%"externalToothed", ext);
  E(e)->addElementText(OPENMBV%"outsideRadius", R);
  return nullptr;
}

void CylindricalGear::initializeUsingXML(DOMElement *element) {
  RigidBody::initializeUsingXML(element);
  DOMElement *e;
  e=E(element)->getFirstElementChildNamed(OPENMBV%"numberOfTeeth");
  setNumberOfTeeth(E(e)->getText<int>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"width");
  setWidth(E(e)->getText<double>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"helixAngle");
  if(e) setHelixAngle(E(e)->getText<double>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"module");
  if(e) setModule(E(e)->getText<double>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"pressureAngle");
  if(e) setPressureAngle(E(e)->getText<double>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"backlash");
  if(e) setBacklash(E(e)->getText<double>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"externalToothed");
  if(e) setExternalToothed(E(e)->getText<bool>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"outsideRadius");
  if(e) setOutsideRadius(E(e)->getText<double>());
}

}
