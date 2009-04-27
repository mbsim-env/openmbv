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
