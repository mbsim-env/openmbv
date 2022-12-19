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
#include <openmbvcppinterface/grid.h>
#include <iostream>
#include <fstream>
#include <boost/lexical_cast.hpp>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(Grid, OPENMBV%"Grid")

Grid::Grid() = default;

DOMElement* Grid::writeXMLFile(DOMNode *parent) {
  DOMElement *e=RigidBody::writeXMLFile(parent);
  E(e)->addElementText(OPENMBV%"xSize", xSize);
  E(e)->addElementText(OPENMBV%"ySize", ySize);
  E(e)->addElementText(OPENMBV%"nx", nx);
  E(e)->addElementText(OPENMBV%"ny", ny);
  return nullptr;
}

void Grid::initializeUsingXML(DOMElement *element) {
  RigidBody::initializeUsingXML(element);
  DOMElement *e;
  e=E(element)->getFirstElementChildNamed(OPENMBV%"xSize");
  setXSize(E(e)->getText<double>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"ySize");
  setYSize(E(e)->getText<double>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"nx");
  setXNumber(boost::lexical_cast<unsigned int>(X()%E(e)->getFirstTextChild()->getData()));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"ny");
  setYNumber(boost::lexical_cast<unsigned int>(X()%E(e)->getFirstTextChild()->getData()));
}

}
