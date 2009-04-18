#ifndef _AMVIS_EXTRUSION_H_
#define _AMVIS_EXTRUSION_H_

#include <amviscppinterface/rigidbody.h>
#include <amviscppinterface/polygonpoint.h>
#include <vector>

namespace AMVis {

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
