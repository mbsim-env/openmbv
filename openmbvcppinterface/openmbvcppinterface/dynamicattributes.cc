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

void DynamicAttributes::setObjectEnable(const PathDataList &p) {
  objectEnable = p;
  updateDataSize();
}

void DynamicAttributes::addObjectEnable(const std::string &p, bool skip) {
  objectEnable.emplace_back(p,skip);
  updateDataSize();
}

void DynamicAttributes::setBodyDrawMethod(const PathDataList &p) {
  bodyDrawMethod = p;
  updateDataSize();
}

void DynamicAttributes::addBodyDrawMethod(const std::string &p, bool skip) {
  bodyDrawMethod.emplace_back(p,skip);
  updateDataSize();
}

void DynamicAttributes::setDynamicColoredBodyTransparency(const PathDataList &p) {
  dynamicColoredBodyTransparency = p;
  updateDataSize();
}

void DynamicAttributes::addDynamicColoredBodyTransparency(const std::string &p, bool skip) {
  dynamicColoredBodyTransparency.emplace_back(p,skip);
  updateDataSize();
}

void DynamicAttributes::updateDataSize() {
  dataSize = 1;

  auto count = [this](const auto &dl){
    for(const auto &d : dl)
      if(!d.skip)
        dataSize++;
  };

  count(objectEnable);
  count(bodyDrawMethod);
  count(dynamicColoredBodyTransparency);
}

DOMElement* DynamicAttributes::writeXMLFile(DOMNode *parent) {
  DOMElement *e=Body::writeXMLFile(parent);

  auto write = [&e](const PathDataList &pdl, const string &xmlName){
    for(auto &pd : pdl) {
      E(e)->addElementText(OPENMBV%xmlName, pd.path);
      if(pd.skip)
        E(e->getLastElementChild())->setAttribute("skip", pd.skip);
    }
  };

  write(objectEnable, "objectEnable");
  write(bodyDrawMethod, "bodyDrawMethod");
  write(dynamicColoredBodyTransparency, "dynamicColoredBodyTransparency");

  return e;
}

void DynamicAttributes::initializeUsingXML(DOMElement *element) {
  Body::initializeUsingXML(element);

  auto read = [&element](PathDataList& pdl, const string &xmlName){
    pdl.clear();
    for(auto e=E(element)->getFirstElementChildNamed(OPENMBV%xmlName); e; e=E(e)->getNextElementSiblingNamed(OPENMBV%xmlName)) {
      auto skipStr = E(e)->getAttribute("skip");
      pdl.emplace_back(E(e)->getText<string>(), skipStr=="1" || skipStr=="true" ? true : false);
    }
  };

  read(objectEnable, "objectEnable");
  read(bodyDrawMethod, "bodyDrawMethod");
  read(dynamicColoredBodyTransparency, "dynamicColoredBodyTransparency");

  updateDataSize();
}

void DynamicAttributes::createHDF5File() {
  Body::createHDF5File();
  data=hdf5Group->createChildObject<H5::VectorSerie<Float> >("data")(dataSize);
  vector<string> columns;
  columns.emplace_back("Time");

  auto create = [&columns](const PathDataList& pdl, const string &name){
    for(auto &pd : pdl)
      if(!pd.skip)
        columns.emplace_back(name+": "+pd.path);
  };

  create(objectEnable, "Object enable");
  create(bodyDrawMethod, "Body drawMethod");
  create(dynamicColoredBodyTransparency, "DynamicColoredBody transparency");

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
