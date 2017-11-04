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
#include <openmbvcppinterface/dynamicindexedfaceset.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(DynamicIndexedFaceSet, OPENMBV%"DynamicIndexedFaceSet")

DynamicIndexedFaceSet::DynamicIndexedFaceSet() : Body(), minimalColorValue(0), maximalColorValue(1), diffuseColor(vector<double>(3)),
  transparency(0), numvp(0) {
  vector<double> hsv(3);
  hsv[0]=-1;
  hsv[1]=1;
  hsv[2]=1;
  diffuseColor=hsv;
}

DOMElement* DynamicIndexedFaceSet::writeXMLFile(DOMNode *parent) {
  DOMElement *e=Body::writeXMLFile(parent);
  E(e)->addElementText(OPENMBV%"minimalColorValue", minimalColorValue);
  E(e)->addElementText(OPENMBV%"maximalColorValue", maximalColorValue);
  if(diffuseColor[0]>=0 || diffuseColor[1]!=1 || diffuseColor[2]!=1)
    E(e)->addElementText(OPENMBV%"diffuseColor", diffuseColor);
  E(e)->addElementText(OPENMBV%"transparency", transparency);
  E(e)->addElementText(OPENMBV%"numberOfVertexPositions", numvp);

  vector<int> indices1based(indices.size());
  transform(indices.begin(), indices.end(), indices1based.begin(), [](int a){ return a+1; });
  E(e)->addElementText(OPENMBV%"indices", indices1based);

  return 0;
}

void DynamicIndexedFaceSet::createHDF5File() {
  Body::createHDF5File();
  if(!hdf5LinkBody) {
    data=hdf5Group->createChildObject<H5::VectorSerie<double> >("data")(1+4*numvp);
    vector<string> columns;
    columns.push_back("Time");
    for(int i=0;i<numvp;i++) {
      columns.push_back("x"+toString(i));
      columns.push_back("y"+toString(i));
      columns.push_back("z"+toString(i));
      columns.push_back("color"+toString(i));
    }
    data->setColumnLabel(columns);
  }
}

void DynamicIndexedFaceSet::openHDF5File() {
  Body::openHDF5File();
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

void DynamicIndexedFaceSet::initializeUsingXML(DOMElement *element) {
  Body::initializeUsingXML(element);
  DOMElement *e=E(element)->getFirstElementChildNamed(OPENMBV%"minimalColorValue");
  if(e)
    setMinimalColorValue(E(e)->getText<double>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"maximalColorValue");
  if(e)
    setMaximalColorValue(E(e)->getText<double>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"diffuseColor");
  if(e)
    setDiffuseColor(E(e)->getText<vector<double>>(3));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"transparency");
  if(e)
    setTransparency(E(e)->getText<double>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"numberOfVertexPositions");
  setNumberOfVertexPositions(E(e)->getText<int>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"indices");
  vector<int> indices1based=E(e)->getText<vector<int>>();
  indices.resize(indices1based.size());
  transform(indices1based.begin(), indices1based.end(), indices.begin(), [](int a){ return a-1; });
}

}
