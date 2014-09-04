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

#include <config.h>
#include "openmbvcppinterface/polygonpoint.h"

using namespace std;

namespace OpenMBV {

void PolygonPoint::serializePolygonPointContour(xercesc::DOMElement *parent, const vector<PolygonPoint*> *cont) {
  string str;
  str="[ ";
  for(vector<PolygonPoint*>::const_iterator j=cont->begin(); j!=cont->end(); j++) {
    str+=Object::numtostr((*j)->getXComponent())+", "+Object::numtostr((*j)->getYComponent())+", "+Object::numtostr((*j)->getBorderValue());
    if(j+1!=cont->end()) str+=";    "; else str+=" ]";
  }
  Object::addElementText(parent, OPENMBV%"contour", str);
}

vector<PolygonPoint*>* PolygonPoint::initializeUsingXML(xercesc::DOMElement *element) {
  vector<vector<double> > matParam=Body::getMat(element);
  vector<PolygonPoint*> *contour=new vector<PolygonPoint*>;
  for(size_t r=0; r<matParam.size(); r++) {
    PolygonPoint *pp=new PolygonPoint(matParam[r][0], matParam[r][1], (int)(matParam[r][2]));
    contour->push_back(pp);
  }
  return contour;
}

}
