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

#ifndef _OPENMBV_EXTRUSION_H_
#define _OPENMBV_EXTRUSION_H_

#include <openmbvcppinterface/rigidbody.h>
#include <openmbvcppinterface/polygonpoint.h>
#include <vector>

namespace OpenMBV {

  /** A extrusion of a cross section area (with holes) */
  class Extrusion : public RigidBody {
    friend class ObjectFactory;
    public:
      enum WindingRule {
        odd,
        nonzero,
        positive,
        negative,
        absGEqTwo
      };
    protected:
      WindingRule windingRule{odd};
      double height{1};
      std::vector<std::shared_ptr<std::vector<std::shared_ptr<PolygonPoint> > > > contour;
      Extrusion();
      ~Extrusion() override;
    public:
      /** Set the OpenGL winding rule for the tesselation of the cross section area.
       * Allowable values are "odd", "nonzero", "positive", "negative" and "absGEqTwo".
       * See the OpenGL-GLU documentation the the meaning of this values.
       */
      void setWindingRule(WindingRule windingRule_) {
        windingRule=windingRule_;
      }

      WindingRule getWindingRule() { return windingRule; }

      /** Set the height of the extrusion.
       * The extrusion is along the normal of the cross section area (local z-axis).
       */
      void setHeight(double height_) {
        height=height_;
      }
      
      double getHeight() { return height; }

      /** Clear all previously added contours. */
      void clearContours() {
        contour.clear();
      }

      /** Add a new contour to the extrusion.
       * See setWindingRule of details about how they are combined.
       */
      void addContour(const std::shared_ptr<std::vector<std::shared_ptr<PolygonPoint> > > &contour_) {
        contour.push_back(contour_);
      }

      std::vector<std::shared_ptr<std::vector<std::shared_ptr<PolygonPoint> > > >& getContours() {
        return contour;
      }

      /** Initializes the time invariant part of the object using a XML node */
      void initializeUsingXML(xercesc::DOMElement *element) override;

      xercesc::DOMElement *writeXMLFile(xercesc::DOMNode *parent) override;

  };

}

#endif /* _OPENMBV_EXTRUSION_H_ */

