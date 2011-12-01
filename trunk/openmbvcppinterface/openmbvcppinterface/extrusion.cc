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
#include <openmbvcppinterface/extrusion.h>
#include <iostream>
#include <fstream>

using namespace std;
using namespace OpenMBV;

Extrusion::Extrusion() : RigidBody(),
  windingRule(odd),
  height(1) {
}

Extrusion::~Extrusion() {
  for(unsigned int i=0;i<contour.size();i++) {
    std::vector<PolygonPoint*> *curContour=contour[i];
    if(curContour) { 
      for(unsigned int i=0;i<curContour->size();i++) {
        delete (*curContour)[i];
        (*curContour)[i]=0;
      }
      delete curContour;
      curContour=0;
    }
  }
}

TiXmlElement *Extrusion::writeXMLFile(TiXmlNode *parent) {
  TiXmlElement *e=RigidBody::writeXMLFile(parent);
  string windingRuleStr;
  switch(windingRule) {
    case odd: windingRuleStr="odd"; break;
    case nonzero: windingRuleStr="nonzero"; break;
    case positive: windingRuleStr="positive"; break;
    case negative: windingRuleStr="negative"; break;
    case absGEqTwo: windingRuleStr="absGEqTwo"; break;
  }
  addElementText(e, "windingRule", "\""+windingRuleStr+"\"");
  addElementText(e, "height", height);
  for(vector<vector<PolygonPoint*>*>::const_iterator i=contour.begin(); i!=contour.end(); i++) 
    PolygonPoint::serializePolygonPointContour(e, *i);
  return 0;
}


void Extrusion::initializeUsingXML(TiXmlElement *element) {
  RigidBody::initializeUsingXML(element);
  TiXmlElement *e;
  e=element->FirstChildElement(OPENMBVNS"windingRule");
  string wrStr=string(e->GetText()).substr(1,string(e->GetText()).length()-2);
  if(wrStr=="odd") setWindingRule(odd);
  if(wrStr=="nonzero") setWindingRule(nonzero);
  if(wrStr=="positive") setWindingRule(positive);
  if(wrStr=="negative") setWindingRule(negative);
  if(wrStr=="absGEqTwo") setWindingRule(absGEqTwo);
  e=element->FirstChildElement(OPENMBVNS"height");
  setHeight(getDouble(e));
  e=e->NextSiblingElement();
  while(e) {
    addContour(PolygonPoint::initializeUsingXML(e));
    e=e->NextSiblingElement();
  }
}
