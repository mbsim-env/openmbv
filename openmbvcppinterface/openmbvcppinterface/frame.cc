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
#include <openmbvcppinterface/frame.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(Frame, OPENMBV%"Frame")

Frame::Frame() : RigidBody(),
  size(1), offset(1) {
}

DOMElement* Frame::writeXMLFile(DOMNode *parent) {
  DOMElement *e=RigidBody::writeXMLFile(parent);
  E(e)->addElementText(OPENMBV%"size", size);
  E(e)->addElementText(OPENMBV%"offset", offset);
  return 0;
}

void Frame::initializeUsingXML(DOMElement *element) {
  RigidBody::initializeUsingXML(element);
  DOMElement *e;
  e=E(element)->getFirstElementChildNamed(OPENMBV%"size");
  setSize(E(e)->getText<double>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"offset");
  setOffset(E(e)->getText<double>());
}

}
