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
#include <openmbvcppinterface/spineextrusion.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(SpineExtrusion, OPENMBV%"SpineExtrusion")

SpineExtrusion::SpineExtrusion() : 
  
  initialRotation(vector<double>(3, 0)) {
}

SpineExtrusion::~SpineExtrusion() = default;

DOMElement* SpineExtrusion::writeXMLFile(DOMNode *parent) {
  DOMElement *e=DynamicColoredBody::writeXMLFile(parent);
  if(contour) PolygonPoint::serializePolygonPointContour(e, contour);
  switch(csOri) {
    case orthogonalWithTwist:
      E(e)->addElementText(OPENMBV%"crossSectionOrientation", "'orthogonalWithTwist'");
      break;
    case cardanWrtWorld:
      E(e)->addElementText(OPENMBV%"crossSectionOrientation", "'cardanWrtWorld'");
      break;
  }
  E(e)->addElementText(OPENMBV%"scaleFactor", scaleFactor);
  E(e)->addElementText(OPENMBV%"initialRotation", initialRotation);
  if( stateOffSet.size() > 0 )
    E(e)->addElementText(OPENMBV%"stateOffSet", vector<double>(stateOffSet));
  return nullptr;
}

void SpineExtrusion::createHDF5File() {
  DynamicColoredBody::createHDF5File();
  vector<string> columns;
  columns.emplace_back("Time");
  switch(csOri) {
    case orthogonalWithTwist:
      data=hdf5Group->createChildObject<H5::VectorSerie<double> >("data")(1+4*numberOfSpinePoints);
      for(int i=0;i<numberOfSpinePoints;i++) {
        columns.push_back("x"+fmatvec::toString(i));
        columns.push_back("y"+fmatvec::toString(i));
        columns.push_back("z"+fmatvec::toString(i));
        columns.push_back("twist"+fmatvec::toString(i));
      }
      break;
    case cardanWrtWorld:
      data=hdf5Group->createChildObject<H5::VectorSerie<double> >("data")(1+6*numberOfSpinePoints);
      for(int i=0;i<numberOfSpinePoints;i++) {
        columns.push_back("x"+fmatvec::toString(i));
        columns.push_back("y"+fmatvec::toString(i));
        columns.push_back("z"+fmatvec::toString(i));
        columns.push_back("alpha"+fmatvec::toString(i));
        columns.push_back("beta"+fmatvec::toString(i));
        columns.push_back("gamma"+fmatvec::toString(i));
      }
      break;
  }
  data->setColumnLabel(columns);
}

void SpineExtrusion::openHDF5File() {
  DynamicColoredBody::openHDF5File();
  if(!hdf5Group) return;
  try {
    data=hdf5Group->openChildObject<H5::VectorSerie<double> >("data");
  }
  catch(...) {
    data=nullptr;
    msg(Debug)<<"Unable to open the HDF5 Dataset 'data'. Using 0 for all data."<<endl;
  }
}

void SpineExtrusion::initializeUsingXML(DOMElement *element) {
  DynamicColoredBody::initializeUsingXML(element);
  DOMElement *e;
  e=E(element)->getFirstElementChildNamed(OPENMBV%"contour");
  setContour(PolygonPoint::initializeUsingXML(e));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"crossSectionOrientation");
  csOri = orthogonalWithTwist;
  if(e) {
    auto cs = E(e)->getText<string>();
    cs = cs.substr(1, cs.length()-2);
    if     (cs == "orthogonalWithTwist") csOri = orthogonalWithTwist;
    else if(cs == "cardanWrtWorld"     ) csOri = cardanWrtWorld;
    else
      throw runtime_error("Unknown 'crossSectionOrientation'");
  }
  e=E(element)->getFirstElementChildNamed(OPENMBV%"scaleFactor");
  setScaleFactor(E(e)->getText<double>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"initialRotation");
  setInitialRotation(E(e)->getText<vector<double>>(3));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"stateOffSet");
  if( e )
    setStateOffSet(E(e)->getText<vector<double>>());
}

}
