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
  E(e)->addElementText(OPENMBV%"dataSize", dataSize);
  E(e)->addElementText(OPENMBV%"scalarData", scalarData);
  if( stateOffSet.size() > 0 )
    E(e)->addElementText(OPENMBV%"stateOffSet", vector<double>(stateOffSet));
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
  dataSize = E(e)->getText<int>();
  e=E(element)->getFirstElementChildNamed(OPENMBV%"scalarData");
  if(e)
    scalarData = E(e)->getText<bool>();
  e=E(element)->getFirstElementChildNamed(OPENMBV%"stateOffSet");
  if( e )
    setStateOffSet(E(e)->getText<vector<double>>());
}

void DynamicIvBody::createHDF5File() {
  Body::createHDF5File();
  data=hdf5Group->createChildObject<H5::VectorSerie<double> >("data")(dataSize);
  vector<string> columns;
  columns.reserve(dataSize);
  for(size_t i=0; i<dataSize; ++i)
    columns.emplace_back("data_"+to_string(i));
  data->setColumnLabel(columns);
}

void DynamicIvBody::openHDF5File() {
  Body::openHDF5File();
  if(!hdf5Group) return;
  try {
    data=hdf5Group->openChildObject<H5::VectorSerie<double> >("data");
  }
  catch(...) {
    data=nullptr;
    msg(Debug)<<"Unable to open the HDF5 Dataset 'data'. Using 0 for all data."<<endl;
  }
  dataSize = data->getColumns();
}

}
