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
#include <openmbvcppinterface/flexiblebody.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

DOMElement* FlexibleBody::writeXMLFile(DOMNode *parent) {
  DOMElement *e=DynamicColoredBody::writeXMLFile(parent);
  E(e)->addElementText(OPENMBV%"numberOfVertexPositions", numvp);
  return e;
}

void FlexibleBody::createHDF5File() {
  DynamicColoredBody::createHDF5File();
  data=hdf5Group->createChildObject<H5::VectorSerie<double> >("data")(1+4*numvp);
  vector<string> columns;
  columns.emplace_back("Time");
  for(int i=0;i<numvp;i++) {
    columns.push_back("x"+fmatvec::toString(i));
    columns.push_back("y"+fmatvec::toString(i));
    columns.push_back("z"+fmatvec::toString(i));
    columns.push_back("color"+fmatvec::toString(i));
  }
  data->setColumnLabel(columns);
}

void FlexibleBody::openHDF5File() {
  Body::openHDF5File();
  if(!hdf5Group) return;
  try {
    data=hdf5Group->openChildObject<H5::VectorSerie<double> >("data");
  }
  catch(...) {
    data=nullptr;
    msg(Warn)<<"Unable to open the HDF5 Dataset 'data'"<<endl;
  }
}

void FlexibleBody::initializeUsingXML(DOMElement *element) {
  DynamicColoredBody::initializeUsingXML(element);
  auto e=E(element)->getFirstElementChildNamed(OPENMBV%"numberOfVertexPositions");
  if(e)
    setNumberOfVertexPositions(E(e)->getText<int>());
}

}
