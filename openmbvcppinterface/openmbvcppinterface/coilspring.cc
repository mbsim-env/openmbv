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

CoilSpring::CoilSpring() : DynamicColoredBody()
  {
}

CoilSpring::~CoilSpring() = default;

DOMElement *CoilSpring::writeXMLFile(DOMNode *parent) {
  DOMElement *e=DynamicColoredBody::writeXMLFile(parent);
  string typeStr;
  switch(type) {
    case tube: typeStr="tube"; break;
    case scaledTube: typeStr="scaledTube"; break;
    case polyline: typeStr="polyline"; break;
  }
  E(e)->addElementText(OPENMBV%"type", "'"+typeStr+"'");
  E(e)->addElementText(OPENMBV%"numberOfCoils", numberOfCoils);
  E(e)->addElementText(OPENMBV%"springRadius", springRadius);
  E(e)->addElementText(OPENMBV%"crossSectionRadius", crossSectionRadius);
  E(e)->addElementText(OPENMBV%"nominalLength", nominalLength);
  E(e)->addElementText(OPENMBV%"scaleFactor", scaleFactor);
  return nullptr;
}

void CoilSpring::createHDF5File() {
  DynamicColoredBody::createHDF5File();
  if(!hdf5LinkBody) {
    data=hdf5Group->createChildObject<H5::VectorSerie<double> >("data")(8);
    vector<string> columns;
    columns.emplace_back("Time");
    columns.emplace_back("fromPoint x");
    columns.emplace_back("fromPoint y");
    columns.emplace_back("fromPoint z");
    columns.emplace_back("toPoint x");
    columns.emplace_back("toPoint y");
    columns.emplace_back("toPoint z");
    columns.emplace_back("color");
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
      data=nullptr;
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
  setNumberOfCoils(E(e)->getText<double>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"springRadius");
  setSpringRadius(E(e)->getText<double>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"crossSectionRadius");
  if(e) setCrossSectionRadius(E(e)->getText<double>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"nominalLength");
  if(e) setNominalLength(E(e)->getText<double>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"scaleFactor");
  if(e) setScaleFactor(E(e)->getText<double>());
}

}
