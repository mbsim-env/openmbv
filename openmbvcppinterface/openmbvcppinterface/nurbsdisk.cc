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
  if(!hdf5LinkBody && data) { delete data; data=0; }
}

DOMElement *NurbsDisk::writeXMLFile(DOMNode *parent) {
  DOMElement *e=DynamicColoredBody::writeXMLFile(parent);
  addElementText(e, OPENMBV%"scaleFactor", scaleFactor);
  addElementText(e, OPENMBV%"drawDegree", drawDegree);
  addElementText(e, OPENMBV%"innerRadius", Ri);
  addElementText(e, OPENMBV%"outerRadius", Ro);
  addElementText(e, OPENMBV%"elementNumberAzimuthal", ElementNumberAzimuthal);
  addElementText(e, OPENMBV%"elementNumberRadial", ElementNumberRadial);
  addElementText(e, OPENMBV%"interpolationDegreeAzimuthal", InterpolationDegreeAzimuthal);
  addElementText(e, OPENMBV%"interpolationDegreeRadial", InterpolationDegreeRadial);
  string str="[";
  for(int i=0;i<getElementNumberAzimuthal()+1+2*getInterpolationDegreeAzimuthal()-1;i++) str+=numtostr(KnotVecAzimuthal[i])+";";
  addElementText(e, OPENMBV%"knotVecAzimuthal", str+numtostr(KnotVecAzimuthal[getElementNumberAzimuthal()+1+2*getInterpolationDegreeAzimuthal()-1])+"]");
  str="[";
  for(int i=0;i<getElementNumberRadial()+1+getInterpolationDegreeRadial();i++) str+=numtostr(KnotVecRadial[i])+";";
  addElementText(e, OPENMBV%"knotVecRadial", str+numtostr(KnotVecRadial[getElementNumberRadial()+1+getInterpolationDegreeRadial()])+"]");

  addAttribute(e, "localFrame", localFrameStr, "false");
  return 0;
}

void NurbsDisk::createHDF5File() {
  DynamicColoredBody::createHDF5File();
  if(!hdf5LinkBody) {
    data=new H5::VectorSerie<double>;
    vector<string> columns;
    int NodeDofs;
    NodeDofs = (getElementNumberRadial() + 1) * (getElementNumberAzimuthal() + getInterpolationDegreeAzimuthal());
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
      columns.push_back("x"+numtostr(i));
      columns.push_back("y"+numtostr(i));
      columns.push_back("z"+numtostr(i));
    }
    for(int i=0;i<getElementNumberAzimuthal()*drawDegree*2;i++) {
      columns.push_back("x"+numtostr(i+NodeDofs));
      columns.push_back("y"+numtostr(i+NodeDofs));
      columns.push_back("z"+numtostr(i+NodeDofs));
    }

    data->create(*hdf5Group,"data",columns);
  }
}

void NurbsDisk::openHDF5File() {
  DynamicColoredBody::openHDF5File();
  if(!hdf5Group) return;
  if(!hdf5LinkBody) {
    data=new H5::VectorSerie<double>;
    try {
      data->open(*hdf5Group,"data");
    }
    catch(...) {
      delete data;
      data=NULL;
      msg(Warn)<<"Unable to open the HDF5 Dataset 'data'"<<endl;
    }
  }
}

void NurbsDisk::initializeUsingXML(DOMElement *element) {
  DynamicColoredBody::initializeUsingXML(element);
  DOMElement *e;
  e=E(element)->getFirstElementChildNamed(OPENMBV%"scaleFactor");
  setScaleFactor(getDouble(e));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"drawDegree");
  setDrawDegree(getDouble(e));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"innerRadius");
  Ri=getDouble(e);
  e=E(element)->getFirstElementChildNamed(OPENMBV%"outerRadius");
  Ro=getDouble(e);
  e=E(element)->getFirstElementChildNamed(OPENMBV%"elementNumberAzimuthal");
  setElementNumberAzimuthal(getDouble(e));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"elementNumberRadial");
  setElementNumberRadial(getDouble(e));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"interpolationDegreeAzimuthal");
  setInterpolationDegreeAzimuthal(getDouble(e));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"interpolationDegreeRadial");
  setInterpolationDegreeRadial(getDouble(e));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"knotVecAzimuthal");
  setKnotVecAzimuthal(getVec(e,getElementNumberAzimuthal()+1+2*getInterpolationDegreeAzimuthal()));
  e=E(element)->getFirstElementChildNamed(OPENMBV%"knotVecRadial");
  setKnotVecRadial(getVec(e,getElementNumberRadial()+1+getInterpolationDegreeRadial()+1));

  if(E(element)->hasAttribute("localFrame") &&
     (E(element)->getAttribute("localFrame")=="true" || E(element)->getAttribute("localFrame")=="1"))
    setLocalFrame(true);
}

}
