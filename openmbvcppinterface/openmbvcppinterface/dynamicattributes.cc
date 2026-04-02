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
#include <openmbvcppinterface/dynamicattributes.h>

using namespace std;
using namespace xercesc;
using namespace MBXMLUtils;

namespace OpenMBV {

OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(DynamicAttributes, OPENMBV%"DynamicAttributes")

DynamicAttributes::DynamicAttributes() = default;

void DynamicAttributes::setObjectEnable(const std::vector<std::string> &p) {
  objectEnable = p;
  updateDataSize();
}

void DynamicAttributes::addObjectEnable(const std::string &p) {
  objectEnable.emplace_back(p);
  updateDataSize();
}

void DynamicAttributes::setBodyDrawMethod(const std::vector<std::string> &p) {
  bodyDrawMethod = p;
  updateDataSize();
}

void DynamicAttributes::addBodyDrawMethod(const std::string &p) {
  bodyDrawMethod.emplace_back(p);
  updateDataSize();
}

void DynamicAttributes::setDynamicColoredBodyTransparency(const std::vector<std::string> &p) {
  dynamicColoredBodyTransparency = p;
  updateDataSize();
}

void DynamicAttributes::addDynamicColoredBodyTransparency(const std::string &p) {
  dynamicColoredBodyTransparency.emplace_back(p);
  updateDataSize();
}

void DynamicAttributes::updateDataSize() {
  dataSize = 1 + objectEnable.size() + bodyDrawMethod.size() + dynamicColoredBodyTransparency.size();
}

DOMElement* DynamicAttributes::writeXMLFile(DOMNode *parent) {
  DOMElement *e=Body::writeXMLFile(parent);
  for(auto &ep : objectEnable)
    E(e)->addElementText(OPENMBV%"objectEnable", ep);
  for(auto &ep : bodyDrawMethod)
    E(e)->addElementText(OPENMBV%"bodyDrawMethod", ep);
  for(auto &ep : dynamicColoredBodyTransparency)
    E(e)->addElementText(OPENMBV%"dynamicColoredBodyTransparency", ep);
  return e;
}

void DynamicAttributes::initializeUsingXML(DOMElement *element) {
  Body::initializeUsingXML(element);

  objectEnable.clear();
  for(auto e=E(element)->getFirstElementChildNamed(OPENMBV%"objectEnable"); e; e=E(e)->getNextElementSiblingNamed(OPENMBV%"objectEnable"))
    objectEnable.emplace_back(E(e)->getText<string>());

  bodyDrawMethod.clear();
  for(auto e=E(element)->getFirstElementChildNamed(OPENMBV%"bodyDrawMethod"); e; e=E(e)->getNextElementSiblingNamed(OPENMBV%"bodyDrawMethod"))
    bodyDrawMethod.emplace_back(E(e)->getText<string>());

  dynamicColoredBodyTransparency.clear();
  for(auto e=E(element)->getFirstElementChildNamed(OPENMBV%"dynamicColoredBodyTransparency"); e; e=E(e)->getNextElementSiblingNamed(OPENMBV%"dynamicColoredBodyTransparency"))
    dynamicColoredBodyTransparency.emplace_back(E(e)->getText<string>());

  updateDataSize();
}

void DynamicAttributes::createHDF5File() {
  Body::createHDF5File();
  data=hdf5Group->createChildObject<H5::VectorSerie<Float> >("data")(dataSize);
  vector<string> columns;
  columns.emplace_back("Time");
  for(auto &p : objectEnable)
    columns.emplace_back("Object enable: "+p);
  for(auto &p : bodyDrawMethod)
    columns.emplace_back("Body drawMethod: "+p);
  for(auto &p : dynamicColoredBodyTransparency)
    columns.emplace_back("DynamicColoredBody transparency: "+p);
  data->setColumnLabel(columns);
}

void DynamicAttributes::openHDF5File() {
  Body::openHDF5File();
  if(!hdf5Group) return;
  try {
    data=hdf5Group->openChildObject<H5::VectorSerie<Float> >("data");
  }
  catch(...) {
    data=nullptr;
    msg(Debug)<<"Unable to open the HDF5 Dataset 'data'. Using 0 for all data."<<endl;
  }
}

}
