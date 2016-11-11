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
#include <openmbvcppinterface/indexeddynamicfaceset.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(IndexedDynamicFaceSet, OPENMBV%"IndexedDynamicFaceSet")

IndexedDynamicFaceSet::IndexedDynamicFaceSet() : DynamicColoredBody(), numvp(0) {
}

DOMElement* IndexedDynamicFaceSet::writeXMLFile(DOMNode *parent) {
  DOMElement *e=DynamicColoredBody::writeXMLFile(parent);
  addElementText(e, OPENMBV%"numberOfVertexPositions", numvp);
  addElementText(e, OPENMBV%"indices", indices);
  return 0;
}

void IndexedDynamicFaceSet::createHDF5File() {
  DynamicColoredBody::createHDF5File();
  if(!hdf5LinkBody) {
    data=hdf5Group->createChildObject<H5::VectorSerie<double> >("data")(1+3*numvp);
    vector<string> columns;
    columns.push_back("Time");
    for(int i=0;i<numvp;i++) {
      columns.push_back("x"+numtostr(i));
      columns.push_back("y"+numtostr(i));
      columns.push_back("z"+numtostr(i));
    }
    data->setColumnLabel(columns);
  }
}

void IndexedDynamicFaceSet::openHDF5File() {
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

void IndexedDynamicFaceSet::initializeUsingXML(DOMElement *element) {
  DynamicColoredBody::initializeUsingXML(element);
  DOMElement *e=E(element)->getFirstElementChildNamed(OPENMBV%"numberOfVertexPositions");
  setNumberOfVertexPositions(getInt(e));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"indices");
  setIndices(getIntVec(e));
}

}
