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
using namespace MBXMLUtils;
using namespace xercesc;

namespace OpenMBV {

OPENMBV_OBJECTFACTORY_REGISTERXMLNAME(Extrusion, OPENMBV%"Extrusion")

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

DOMElement *Extrusion::writeXMLFile(DOMNode *parent) {
  DOMElement *e=RigidBody::writeXMLFile(parent);
  string windingRuleStr;
  switch(windingRule) {
    case odd: windingRuleStr="odd"; break;
    case nonzero: windingRuleStr="nonzero"; break;
    case positive: windingRuleStr="positive"; break;
    case negative: windingRuleStr="negative"; break;
    case absGEqTwo: windingRuleStr="absGEqTwo"; break;
  }
  addElementText(e, OPENMBV%"windingRule", "\""+windingRuleStr+"\"");
  addElementText(e, OPENMBV%"height", height);
  for(vector<vector<PolygonPoint*>*>::const_iterator i=contour.begin(); i!=contour.end(); i++) 
    PolygonPoint::serializePolygonPointContour(e, *i);
  return 0;
}


void Extrusion::initializeUsingXML(DOMElement *element) {
  RigidBody::initializeUsingXML(element);
  DOMElement *e;
  e=E(element)->getFirstElementChildNamed(OPENMBV%"windingRule");
  string text = X()%E(e)->getFirstTextChild()->getData();
  string wrStr=text.substr(1,text.length()-2);
  if(wrStr=="odd") setWindingRule(odd);
  if(wrStr=="nonzero") setWindingRule(nonzero);
  if(wrStr=="positive") setWindingRule(positive);
  if(wrStr=="negative") setWindingRule(negative);
  if(wrStr=="absGEqTwo") setWindingRule(absGEqTwo);
  e=E(element)->getFirstElementChildNamed(OPENMBV%"height");
  setHeight(getDouble(e));
  e=e->getNextElementSibling();
  while(e) {
    addContour(PolygonPoint::initializeUsingXML(e));
    e=e->getNextElementSibling();
  }
}

}
