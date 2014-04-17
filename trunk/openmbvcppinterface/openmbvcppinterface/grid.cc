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
#include <openmbvcppinterface/grid.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(Grid, OPENMBV%"Grid")

Grid::Grid() : RigidBody(),
  xSize(1), ySize(1), nx(10), ny(10) {
}

DOMElement* Grid::writeXMLFile(DOMNode *parent) {
  DOMElement *e=RigidBody::writeXMLFile(parent);
  addElementText(e, OPENMBV%"xSize", xSize);
  addElementText(e, OPENMBV%"ySize", ySize);
  addElementText(e, OPENMBV%"nx", nx);
  addElementText(e, OPENMBV%"ny", ny);
  return 0;
}

void Grid::initializeUsingXML(DOMElement *element) {
  RigidBody::initializeUsingXML(element);
  DOMElement *e;
  e=E(element)->getFirstElementChildNamed(OPENMBV%"xSize");
  setXSize(getDouble(e));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"ySize");
  setYSize(getDouble(e));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"nx");
  setXNumber((unsigned int)(atof((X()%E(e)->getFirstTextChild()->getData()).c_str())+.1));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"ny");
  setYNumber((unsigned int)(atof((X()%E(e)->getFirstTextChild()->getData()).c_str())+.1));
}

}
