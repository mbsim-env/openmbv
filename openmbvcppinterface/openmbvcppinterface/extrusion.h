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

#ifndef _OPENMBV_EXTRUSION_H_
#define _OPENMBV_EXTRUSION_H_

#include <openmbvcppinterface/rigidbody.h>
#include <openmbvcppinterface/polygonpoint.h>
#include <vector>

namespace OpenMBV {

  /** A extrusion of a cross section area (with holes) */
  class Extrusion : public RigidBody {
    protected:
      enum WindingRule {
        odd,
        nonzero,
        positive,
        negative,
        absGEqTwo
      };
      WindingRule windingRule;
      double height;
      std::vector<std::vector<PolygonPoint*>*> contour;
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
    public:
      /** Default constructor */
      Extrusion();

      /** Set the OpenGL winding rule for the tesselation of the crsoss section area.
       * Allowable values are "odd", "nonzero", "positive", "negative" and "absGEqTwo".
       * See the OpenGL-GLU documentation the the meaning of this values.
       */
      void setWindingRule(WindingRule windingRule_) {
        windingRule=windingRule_;
      }

      /** Set the height of the extrusion.
       * The extrusion is along the normal of the cross section area (local z-axis).
       */
      void setHeight(float height_) {
        height=height_;
      }

      /** Clear all previously added contours. */
      void clearContours() {
        contour.clear();
      }

      /** Add a new contour to the extrusion.
       * See setWindingRule of details about how they are combined.
       */
      void addContour(std::vector<PolygonPoint*> *contour_) {
        contour.push_back(contour_);
      }

  };

}

#endif
