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
#include <openmbvcppinterface/coilspring.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(CoilSpring, OPENMBV%"CoilSpring")

CoilSpring::CoilSpring() : DynamicColoredBody(),
  data(0),
  springRadius(1),
  crossSectionRadius(-1),
  scaleFactor(1),
  numberOfCoils(3),
  nominalLength(-1),
  type(tube) {
}

CoilSpring::~CoilSpring() {
}

DOMElement *CoilSpring::writeXMLFile(DOMNode *parent) {
  DOMElement *e=DynamicColoredBody::writeXMLFile(parent);
  string typeStr;
  switch(type) {
    case tube: typeStr="tube"; break;
    case scaledTube: typeStr="scaledTube"; break;
    case polyline: typeStr="polyline"; break;
  }
  addElementText(e, OPENMBV%"type", "'"+typeStr+"'");
  addElementText(e, OPENMBV%"numberOfCoils", numberOfCoils);
  addElementText(e, OPENMBV%"springRadius", springRadius);
  addElementText(e, OPENMBV%"crossSectionRadius", crossSectionRadius);
  addElementText(e, OPENMBV%"nominalLength", nominalLength);
  addElementText(e, OPENMBV%"scaleFactor", scaleFactor);
  return 0;
}

void CoilSpring::createHDF5File() {
  DynamicColoredBody::createHDF5File();
  if(!hdf5LinkBody) {
    data=hdf5Group->createChildObject<H5::VectorSerie<double> >("data")(8);
    vector<string> columns;
    columns.push_back("Time");
    columns.push_back("fromPoint x");
    columns.push_back("fromPoint y");
    columns.push_back("fromPoint z");
    columns.push_back("toPoint x");
    columns.push_back("toPoint y");
    columns.push_back("toPoint z");
    columns.push_back("color");
    data->setColumnLabel(columns);
  }
}

void CoilSpring::openHDF5File() {
  DynamicColoredBody::openHDF5File();
  if(!hdf5Group) return;
  if(!hdf5LinkBody) {
    try {
      data=hdf5Group->openChildObject<H5::VectorSerie<double> >("data");
    }
    catch(...) {
      data=NULL;
      msg(Warn)<<"Unable to open the HDF5 Dataset 'data'"<<endl;
    }
  }
}

void CoilSpring::initializeUsingXML(DOMElement *element) {
  DynamicColoredBody::initializeUsingXML(element);
  DOMElement *e;
  e=E(element)->getFirstElementChildNamed(OPENMBV%"type");
  if(e) {
    string name = X()%E(e)->getFirstTextChild()->getData();
    string typeStr=name.substr(1,name.length()-2);
    if(typeStr=="tube") setType(tube);
    if(typeStr=="scaledTube") setType(scaledTube);
    if(typeStr=="polyline") setType(polyline);
  }
  e=E(element)->getFirstElementChildNamed(OPENMBV%"numberOfCoils");
  setNumberOfCoils(getDouble(e));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"springRadius");
  setSpringRadius(getDouble(e));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"crossSectionRadius");
  if(e) setCrossSectionRadius(getDouble(e));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"nominalLength");
  if(e) setNominalLength(getDouble(e));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"scaleFactor");
  if(e) setScaleFactor(getDouble(e));
}

}
