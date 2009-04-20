#ifndef _OPENMBV_EXTRUSION_H_
#define _OPENMBV_EXTRUSION_H_

#include <openmbvcppinterface/rigidbody.h>
#include <openmbvcppinterface/polygonpoint.h>
#include <vector>

namespace OpenMBV {

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
      Extrusion();
      void setWindingRule(WindingRule windingRule_) {
        windingRule=windingRule_;
      }
      void setHeight(float height_) {
        height=height_;
      }
      void clearContours() {
        contour.clear();
      }
      void addContour(std::vector<PolygonPoint*> *contour_) {
        contour.push_back(contour_);
      }

  };

}

#endif
