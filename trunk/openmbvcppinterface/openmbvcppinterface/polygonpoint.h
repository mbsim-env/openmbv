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
      PolygonPoint(DoubleParam x_, DoubleParam y_, int b_) : x(x_), y(y_), b(b_) {}

      /* GETTER / SETTER */
      DoubleParam getXComponent() { return x; } 
      DoubleParam getYComponent() { return y; } 
      int getBorderValue() { return b; }
      /***************************************************/

      /* CONVENIENCE */
      /** write vector of polygon points to XML file */
      static void serializePolygonPointContour(std::ofstream& xmlFile, const std::string& indent, const std::vector<PolygonPoint*> *cont) {
        xmlFile<<indent<<"<contour>"<<std::endl;
        xmlFile<<indent<<"  [ ";
        for(std::vector<PolygonPoint*>::const_iterator j=cont->begin(); j!=cont->end(); j++) {
          if(j!=cont->begin()) xmlFile<<indent<<"    ";
          xmlFile<<(*j)->getXComponent() <<", "<<(*j)->getYComponent() <<", "<<(*j)->getBorderValue();
          if(j+1!=cont->end()) xmlFile<<";"<<std::endl; else xmlFile<<" ]"<<std::endl;
        }
        xmlFile<<indent<<"</contour>"<<std::endl;
      }

      static std::vector<PolygonPoint*>* initializeUsingXML(TiXmlElement *element) {
        std::vector<std::vector<double> > mat=Body::getMat(element);
        std::vector<PolygonPoint*> *contour=new std::vector<PolygonPoint*>;
        for(size_t r=0; r<mat.size(); r++) {
          PolygonPoint *pp=new PolygonPoint(mat[r][0], mat[r][1], (int)(mat[r][2]));
          contour->push_back(pp);
        }
        return contour;
      }

    private:
      DoubleParam x, y;
      int b;
  };

}

#endif /* POLYGONPOINT_H */

