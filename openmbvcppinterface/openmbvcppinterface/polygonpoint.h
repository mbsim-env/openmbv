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

#ifndef POLYGONPOINT_H
#define POLYGONPOINT_H

#include <vector>
#include <fstream>
#include <iostream>
#include "openmbvcppinterfacetinyxml/tinyxml-src/tinyxml.h"
#include "openmbvcppinterface/body.h"

namespace OpenMBV {

  /*!
   * Polygon point
   * x and y are the coordinates of a polygon-edge. If b is 0 this
   * edge is in reality not a edge and is rendered smooth in OpenMBV. If b is 1
   * this edge is rendered non-smooth in OpenMBV.
   */
  class PolygonPoint {
    public:
      /** constructor */
      PolygonPoint(double x_, double y_, int b_) : x(x_), y(y_), b(b_) {}

      /* GETTER / SETTER */
      double getXComponent() { return x; } 
      double getYComponent() { return y; } 
      int getBorderValue() { return b; }
      /***************************************************/

      /* CONVENIENCE */
      /** write vector of polygon points to XML file */
      static void serializePolygonPointContour(TiXmlElement *parent, const std::vector<PolygonPoint*> *cont) {
        std::string str;
        str="[ ";
        for(std::vector<PolygonPoint*>::const_iterator j=cont->begin(); j!=cont->end(); j++) {
          str+=Object::numtostr((*j)->getXComponent())+", "+Object::numtostr((*j)->getYComponent())+", "+Object::numtostr((*j)->getBorderValue());
          if(j+1!=cont->end()) str+=";    "; else str+=" ]";
        }
        Object::addElementText(parent, OPENMBVNS"contour", str);
      }

      static std::vector<PolygonPoint*>* initializeUsingXML(TiXmlElement *element) {
        MatrixParameter matParam=Body::getMat(element);
        assert(matParam.getParamStr()=="" && "Only numeric values are allowd for contours (vector<PolygonPoint*>)");
        std::vector<std::vector<double> > mat=matParam.getValue();
        std::vector<PolygonPoint*> *contour=new std::vector<PolygonPoint*>;
        for(size_t r=0; r<mat.size(); r++) {
          PolygonPoint *pp=new PolygonPoint(mat[r][0], mat[r][1], (int)(mat[r][2]));
          contour->push_back(pp);
        }
        return contour;
      }

    private:
      double x, y;
      int b;
  };

}

#endif /* POLYGONPOINT_H */

