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
#include <openmbvcppinterface/dynamicivbody.h>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(DynamicIvBody, OPENMBV%"DynamicIvBody")

DynamicIvBody::DynamicIvBody() = default;

DOMElement* DynamicIvBody::writeXMLFile(DOMNode *parent) {
  DOMElement *e=Body::writeXMLFile(parent);
  if(!ivFileName.empty())
    E(e)->addElementText(OPENMBV%"ivFileName", "'"+ivFileName+"'");
  else
    E(e)->addElementText(OPENMBV%"ivContent", "'"+ivContent+"'");
  if(dataSize>0)
    E(e)->addElementText(OPENMBV%"dataSize", dataSize);
  if(dataIntSize>0)
    E(e)->addElementText(OPENMBV%"dataIntSize", dataIntSize);
  if(dataStrSize>0)
    E(e)->addElementText(OPENMBV%"dataStrSize", dataStrSize);
  E(e)->addElementText(OPENMBV%"scalarData", scalarData);
  if( stateOffSet.size() > 0 )
    E(e)->addElementText(OPENMBV%"stateOffSet", vector<Float>(stateOffSet));
  return nullptr;
}

void DynamicIvBody::initializeUsingXML(DOMElement *element) {
  Body::initializeUsingXML(element);
  DOMElement *e;
  e=E(element)->getFirstElementChildNamed(OPENMBV%"ivFileName");
  if(e) {
    string str = X()%E(e)->getFirstTextChild()->getData();
    setIvFileName(E(e)->convertPath(str.substr(1,str.length()-2)).string());
  }
  e=E(element)->getFirstElementChildNamed(OPENMBV%"ivContent");
  if(e) {
    string str = X()%E(e)->getFirstTextChild()->getData();
    setIvContent(str.substr(1,str.length()-2));
  }
  e=E(element)->getFirstElementChildNamed(OPENMBV%"dataSize");
  dataSize = 0;
  if(e)
    dataSize = E(e)->getText<int>();
  e=E(element)->getFirstElementChildNamed(OPENMBV%"dataIntSize");
  dataIntSize = 0;
  if(e)
    dataIntSize = E(e)->getText<int>();
  e=E(element)->getFirstElementChildNamed(OPENMBV%"dataStrSize");
  dataStrSize = 0;
  if(e)
    dataStrSize = E(e)->getText<int>();
  e=E(element)->getFirstElementChildNamed(OPENMBV%"scalarData");
  if(e)
    scalarData = E(e)->getText<bool>();
  e=E(element)->getFirstElementChildNamed(OPENMBV%"stateOffSet");
  if( e )
    setStateOffSet(E(e)->getText<vector<double>>());
}

void DynamicIvBody::createHDF5File() {
  Body::createHDF5File();

  auto create = [this](int size, auto &data, const string &name){
    if(size>0) {
      data=hdf5Group->createChildObject<std::remove_pointer_t<std::remove_reference_t<decltype(data)>>>(name)(size);
      vector<string> columns;
      columns.reserve(size);
      for(int i=0; i<size; ++i)
        columns.emplace_back(name+"_"+to_string(i));
      data->setColumnLabel(columns);
    }
  };

  create(dataSize   , data   , "data");
  create(dataIntSize, dataInt, "dataInt");
  create(dataStrSize, dataStr, "dataStr");
}

void DynamicIvBody::openHDF5File() {
  Body::openHDF5File();
  if(!hdf5Group) return;

  auto open = [this](size_t &size, auto &data, const string &name){
    try {
      data=hdf5Group->openChildObject<std::remove_pointer_t<std::remove_reference_t<decltype(data)>>>(name);
    }
    catch(...) {
      data=nullptr;
      size = 0;
    }
    size = data->getColumns();
  };
  open(dataSize   , data   , "data");
  open(dataIntSize, dataInt, "dataInt");
  open(dataStrSize, dataStr, "dataStr");

}

}
