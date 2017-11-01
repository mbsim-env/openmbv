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
#include <openmbvcppinterface/nurbsdisk.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(NurbsDisk, OPENMBV%"NurbsDisk")

NurbsDisk::NurbsDisk() : DynamicColoredBody(),
  data(0),
  localFrameStr("false"),
  scaleFactor(1),
  drawDegree(1),
  Ri(0.),
  Ro(0.),
  ElementNumberAzimuthal(0),
  ElementNumberRadial(0),
  InterpolationDegreeAzimuthal(8),
  InterpolationDegreeRadial(3),
  KnotVecAzimuthal(vector<double>(17,0)),
  KnotVecRadial(vector<double>(4,0)),
  DiskNormal(0),
  DiskPoint(0) {
  }

NurbsDisk::~NurbsDisk() {
}

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
  for(int i=0;i<getElementNumberAzimuthal()+1+2*getInterpolationDegreeAzimuthal()-1;i++) str+=to_string(KnotVecAzimuthal[i])+";";
  E(e)->addElementText(OPENMBV%"knotVecAzimuthal", str+to_string(KnotVecAzimuthal[getElementNumberAzimuthal()+1+2*getInterpolationDegreeAzimuthal()-1])+"]");
  str="[";
  for(int i=0;i<getElementNumberRadial()+1+getInterpolationDegreeRadial();i++) str+=to_string(KnotVecRadial[i])+";";
  E(e)->addElementText(OPENMBV%"knotVecRadial", str+to_string(KnotVecRadial[getElementNumberRadial()+1+getInterpolationDegreeRadial()])+"]");

  E(e)->setAttribute("localFrame", localFrameStr);
  return 0;
}

void NurbsDisk::createHDF5File() {
  DynamicColoredBody::createHDF5File();
  if(!hdf5LinkBody) {
    int NodeDofs;
    NodeDofs = (getElementNumberRadial() + 1) * (getElementNumberAzimuthal() + getInterpolationDegreeAzimuthal());
    data=hdf5Group->createChildObject<H5::VectorSerie<double> >("data")(7+3*NodeDofs+3*getElementNumberAzimuthal()*drawDegree*2);
    vector<string> columns;
    columns.push_back("Time");

    //Global position (position of center of gravity)
    columns.push_back("Pos_x");
    columns.push_back("Pos_y");
    columns.push_back("Pos_z");

    //Global orientation (= Cardan angles for orientation of COG)
    columns.push_back("Rot_alpha");
    columns.push_back("Rot_beta");
    columns.push_back("Rot_gamma");

    //coordinates of control points
    for(int i=0;i<NodeDofs;i++) {
      columns.push_back("x"+to_string(i));
      columns.push_back("y"+to_string(i));
      columns.push_back("z"+to_string(i));
    }
    for(int i=0;i<getElementNumberAzimuthal()*drawDegree*2;i++) {
      columns.push_back("x"+to_string(i+NodeDofs));
      columns.push_back("y"+to_string(i+NodeDofs));
      columns.push_back("z"+to_string(i+NodeDofs));
    }

    data->setColumnLabel(columns);
  }
}

void NurbsDisk::openHDF5File() {
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
