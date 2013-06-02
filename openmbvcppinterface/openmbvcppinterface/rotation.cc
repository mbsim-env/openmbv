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
#include <openmbvcppinterface/rotation.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace MBXMLUtils;
using namespace OpenMBV;

OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(Rotation, OPENMBVNS"Rotation")

Rotation::Rotation() : RigidBody(),
  startAngle(0),
  endAngle(2*M_PI),
  contour(0) {
}

Rotation::~Rotation() {
  if(contour) { 
    for(unsigned int i=0;i<contour->size();i++) {
      delete (*contour)[i];
      (*contour)[i]=0;
    }
    delete contour;
    contour=0;
  }
}

TiXmlElement* Rotation::writeXMLFile(TiXmlNode *parent) {
  TiXmlElement *e=RigidBody::writeXMLFile(parent);
  addElementText(e, OPENMBVNS"startAngle", startAngle, 0);
  addElementText(e, OPENMBVNS"endAngle", endAngle, 2*M_PI);
  if(contour) PolygonPoint::serializePolygonPointContour(e, contour);
  return 0;
}


void Rotation::initializeUsingXML(TiXmlElement *element) {
  RigidBody::initializeUsingXML(element);
  TiXmlElement *e;
  e=element->FirstChildElement(OPENMBVNS"startAngle");
  if(e) setStartAngle(getDouble(e));
  e=element->FirstChildElement(OPENMBVNS"endAngle");
  if(e) setEndAngle(getDouble(e));
  e=element->FirstChildElement(OPENMBVNS"contour");
  setContour(PolygonPoint::initializeUsingXML(e));
}
