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
#include <openmbvcppinterface/dynamicnurbssurface.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(DynamicNurbsSurface, OPENMBV%"DynamicNurbsSurface")

DOMElement* DynamicNurbsSurface::writeXMLFile(DOMNode *parent) {
  DOMElement *e=DynamicColoredBody::writeXMLFile(parent);
  E(e)->addElementText(OPENMBV%"numberOfUControlPoints", numU);
  E(e)->addElementText(OPENMBV%"numberOfVControlPoints", numV);
  E(e)->addElementText(OPENMBV%"uKnotVector", uKnot);
  E(e)->addElementText(OPENMBV%"vKnotVector", vKnot);
  return nullptr;
}

void DynamicNurbsSurface::createHDF5File() {
  DynamicColoredBody::createHDF5File();
  if(!hdf5LinkBody) {
    data=hdf5Group->createChildObject<H5::VectorSerie<double> >("data")(1+4*numU*numV);
    vector<string> columns;
    columns.emplace_back("Time");
    for(int i=0;i<numU*numV;i++) {
      columns.push_back("x"+fmatvec::toString(i));
      columns.push_back("y"+fmatvec::toString(i));
      columns.push_back("z"+fmatvec::toString(i));
      columns.push_back("w"+fmatvec::toString(i));
    }
    data->setColumnLabel(columns);
  }
}

void DynamicNurbsSurface::openHDF5File() {
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

void DynamicNurbsSurface::initializeUsingXML(DOMElement *element) {
  DynamicColoredBody::initializeUsingXML(element);
  DOMElement *e=E(element)->getFirstElementChildNamed(OPENMBV%"numberOfUControlPoints");
  setNumberOfUControlPoints(E(e)->getText<int>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"numberOfVControlPoints");
  setNumberOfVControlPoints(E(e)->getText<int>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"uKnotVector");
  setUKnotVector(E(e)->getText<vector<double>>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"vKnotVector");
  setVKnotVector(E(e)->getText<vector<double>>());
}

}
