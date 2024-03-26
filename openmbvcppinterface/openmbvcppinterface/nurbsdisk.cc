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
#include <openmbvcppinterface/nurbsdisk.h>
#include <iostream>
#include <fstream>
#include <fmatvec/toString.h>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(NurbsDisk, OPENMBV%"NurbsDisk")

NurbsDisk::NurbsDisk() : 
  
  localFrameStr("false"),
  
  KnotVecAzimuthal(vector<double>(17,0)),
  KnotVecRadial(vector<double>(4,0))
  {
  }

NurbsDisk::~NurbsDisk() = default;

DOMElement *NurbsDisk::writeXMLFile(DOMNode *parent) {
  DOMElement *e=DynamicColoredBody::writeXMLFile(parent);
  E(e)->addElementText(OPENMBV%"scaleFactor", scaleFactor);
  E(e)->addElementText(OPENMBV%"drawDegree", drawDegree);
  E(e)->addElementText(OPENMBV%"innerRadius", Ri);
  E(e)->addElementText(OPENMBV%"outerRadius", Ro);
  E(e)->addElementText(OPENMBV%"elementNumberAzimuthal", ElementNumberAzimuthal);
  E(e)->addElementText(OPENMBV%"elementNumberRadial", ElementNumberRadial);
  E(e)->addElementText(OPENMBV%"interpolationDegreeAzimuthal", InterpolationDegreeAzimuthal);
  E(e)->addElementText(OPENMBV%"interpolationDegreeRadial", InterpolationDegreeRadial);
  string str="[";
  for(int i=0;i<getElementNumberAzimuthal()+1+2*getInterpolationDegreeAzimuthal()-1;i++) str+=fmatvec::toString(KnotVecAzimuthal[i])+";";
  E(e)->addElementText(OPENMBV%"knotVecAzimuthal", str+fmatvec::toString(KnotVecAzimuthal[getElementNumberAzimuthal()+1+2*getInterpolationDegreeAzimuthal()-1])+"]");
  str="[";
  for(int i=0;i<getElementNumberRadial()+1+getInterpolationDegreeRadial();i++) str+=fmatvec::toString(KnotVecRadial[i])+";";
  E(e)->addElementText(OPENMBV%"knotVecRadial", str+fmatvec::toString(KnotVecRadial[getElementNumberRadial()+1+getInterpolationDegreeRadial()])+"]");

  E(e)->setAttribute("localFrame", localFrameStr);
  return nullptr;
}

void NurbsDisk::createHDF5File() {
  DynamicColoredBody::createHDF5File();
  int NodeDofs;
  NodeDofs = (getElementNumberRadial() + 1) * (getElementNumberAzimuthal() + getInterpolationDegreeAzimuthal());
  data=hdf5Group->createChildObject<H5::VectorSerie<double> >("data")(7+3*NodeDofs+3*getElementNumberAzimuthal()*drawDegree*2);
  vector<string> columns;
  columns.emplace_back("Time");

  //Global position (position of center of gravity)
  columns.emplace_back("Pos_x");
  columns.emplace_back("Pos_y");
  columns.emplace_back("Pos_z");

  //Global orientation (= Cardan angles for orientation of COG)
  columns.emplace_back("Rot_alpha");
  columns.emplace_back("Rot_beta");
  columns.emplace_back("Rot_gamma");

  //coordinates of control points
  for(int i=0;i<NodeDofs;i++) {
    columns.push_back("x"+fmatvec::toString(i));
    columns.push_back("y"+fmatvec::toString(i));
    columns.push_back("z"+fmatvec::toString(i));
  }
  for(int i=0;i<getElementNumberAzimuthal()*drawDegree*2;i++) {
    columns.push_back("x"+fmatvec::toString(i+NodeDofs));
    columns.push_back("y"+fmatvec::toString(i+NodeDofs));
    columns.push_back("z"+fmatvec::toString(i+NodeDofs));
  }

  data->setColumnLabel(columns);
}

void NurbsDisk::openHDF5File() {
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

void NurbsDisk::initializeUsingXML(DOMElement *element) {
  DynamicColoredBody::initializeUsingXML(element);
  DOMElement *e;
  e=E(element)->getFirstElementChildNamed(OPENMBV%"scaleFactor");
  setScaleFactor(E(e)->getText<double>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"drawDegree");
  setDrawDegree(E(e)->getText<double>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"innerRadius");
  Ri=E(e)->getText<double>();
  e=E(element)->getFirstElementChildNamed(OPENMBV%"outerRadius");
  Ro=E(e)->getText<double>();
  e=E(element)->getFirstElementChildNamed(OPENMBV%"elementNumberAzimuthal");
  setElementNumberAzimuthal(E(e)->getText<double>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"elementNumberRadial");
  setElementNumberRadial(E(e)->getText<double>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"interpolationDegreeAzimuthal");
  setInterpolationDegreeAzimuthal(E(e)->getText<double>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"interpolationDegreeRadial");
  setInterpolationDegreeRadial(E(e)->getText<double>());
  e=E(element)->getFirstElementChildNamed(OPENMBV%"knotVecAzimuthal");
  setKnotVecAzimuthal(E(e)->getText<vector<double>>(getElementNumberAzimuthal()+1+2*getInterpolationDegreeAzimuthal()));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"knotVecRadial");
  setKnotVecRadial(E(e)->getText<vector<double>>(getElementNumberRadial()+1+getInterpolationDegreeRadial()+1));

  if(E(element)->hasAttribute("localFrame") &&
     (E(element)->getAttribute("localFrame")=="true" || E(element)->getAttribute("localFrame")=="1"))
    setLocalFrame(true);
}

}
