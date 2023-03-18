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
#include <openmbvcppinterface/path.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(Path, OPENMBV%"Path")

Path::Path() :  data(nullptr), color(vector<double>(3,1)) {
}

Path::~Path() = default;

DOMElement* Path::writeXMLFile(DOMNode *parent) {
  DOMElement *e=Body::writeXMLFile(parent);
  E(e)->addElementText(OPENMBV%"color", color);
  return nullptr;
}

void Path::createHDF5File() {
  Body::createHDF5File();
  data=hdf5Group->createChildObject<H5::VectorSerie<double> >("data")(4);
  vector<string> columns;
  columns.emplace_back("Time");
  columns.emplace_back("x");
  columns.emplace_back("y");
  columns.emplace_back("z");
  data->setColumnLabel(columns);
}

void Path::openHDF5File() {
  Body::openHDF5File();
  if(!hdf5Group) return;
  try {
    data=hdf5Group->openChildObject<H5::VectorSerie<double> >("data");
  }
  catch(...) {
    data=nullptr;
    msg(Info)<<"Unable to open the HDF5 Dataset 'data'. Using 0 for all data."<<endl;
  }
}

void Path::initializeUsingXML(DOMElement *element) {
  Body::initializeUsingXML(element);
  DOMElement *e;
  e=E(element)->getFirstElementChildNamed(OPENMBV%"color");
  setColor(E(e)->getText<vector<double>>(3));
}

}
