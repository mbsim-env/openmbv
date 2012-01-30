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
using namespace OpenMBV;

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

TiXmlElement *NurbsDisk::writeXMLFile(TiXmlNode *parent) {
  TiXmlElement *e=DynamicColoredBody::writeXMLFile(parent);
  addElementText(e, "localFrame", localFrameStr);
  addElementText(e, "scaleFactor", scaleFactor);
  addElementText(e, "drawDegree", drawDegree);
  addElementText(e, "innerRadius", Ri);
  addElementText(e, "outerRadius", Ro);
  addElementText(e, "elementNumberAzimuthal", ElementNumberAzimuthal);
  addElementText(e, "elementNumberRadial", ElementNumberRadial);
  addElementText(e, "interpolationDegreeAzimuthal", InterpolationDegreeAzimuthal);
  addElementText(e, "interpolationDegreeRadial", InterpolationDegreeRadial);
  string str="[";
  for(int i=0;i<getElementNumberAzimuthal()+1+2*getInterpolationDegreeAzimuthal()-1;i++) str+=numtostr(KnotVecAzimuthal.getValue()[i])+";";
  addElementText(e, "knotVecAzimuthal", str+numtostr(KnotVecAzimuthal.getValue()[getElementNumberAzimuthal()+1+2*getInterpolationDegreeAzimuthal()-1])+"]");
  str="[";
  for(int i=0;i<getElementNumberRadial()+1+getInterpolationDegreeRadial();i++) str+=numtostr(KnotVecRadial.getValue()[i])+";";
  addElementText(e, "knotVecRadial", str+numtostr(KnotVecRadial.getValue()[getElementNumberRadial()+1+getInterpolationDegreeRadial()])+"]");
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
    for(int i=0;i<getElementNumberAzimuthal()*get(drawDegree)*2;i++) {
      columns.push_back("x"+numtostr(i+NodeDofs));
      columns.push_back("y"+numtostr(i+NodeDofs));
      columns.push_back("z"+numtostr(i+NodeDofs));
    }

    data->create(*hdf5Group,"data",columns);
  }
}

void NurbsDisk::openHDF5File() {
  DynamicColoredBody::openHDF5File();
  if(!hdf5LinkBody) {
    data=new H5::VectorSerie<double>;
    data->open(*hdf5Group,"data");
  }
}

void NurbsDisk::initializeUsingXML(TiXmlElement *element) {
  DynamicColoredBody::initializeUsingXML(element);
  if(element->Attribute("localFrame") &&
     (element->Attribute("localFrame")==string("true") || element->Attribute("localFrame")==string("1")))
    setLocalFrame(true);

  TiXmlElement *e;
  e=element->FirstChildElement(OPENMBVNS"scaleFactor");
  setScaleFactor(getDouble(e));
  addElementText(e, "drawDegree", drawDegree);
  setDrawDegree(getDouble(e));
  e=element->FirstChildElement(OPENMBVNS"innerRadius");
  set(Ri,getDouble(e));
  e=element->FirstChildElement(OPENMBVNS"outerRadius");
  set(Ro,getDouble(e));
  e=element->FirstChildElement(OPENMBVNS"elementNumberAzimuthal");
  setElementNumberAzimuthal(getDouble(e));
  e=element->FirstChildElement(OPENMBVNS"elementNumberRadial");
  setElementNumberRadial(getDouble(e));
  e=element->FirstChildElement(OPENMBVNS"interpolationDegreeAzimuthal");
  setInterpolationDegreeAzimuthal(getDouble(e));
  e=element->FirstChildElement(OPENMBVNS"interpolationDegreeRadial");
  setInterpolationDegreeRadial(getDouble(e));
  e=element->FirstChildElement(OPENMBVNS"knotVecAzimuthal");
  setKnotVecAzimuthal(getVec(e,getElementNumberAzimuthal()+1+2*getInterpolationDegreeAzimuthal()));
  e=element->FirstChildElement(OPENMBVNS"knotVecRadial");
  setKnotVecRadial(getVec(e,getElementNumberRadial()+1+getInterpolationDegreeRadial()+1));
}

