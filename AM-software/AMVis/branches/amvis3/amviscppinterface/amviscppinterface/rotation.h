#ifndef _AMVIS_ROTATION_H_
#define _AMVIS_ROTATION_H_

#include <amviscppinterface/rigidbody.h>
#include <amviscppinterface/polygonpoint.h>
#include <vector>

namespace AMVis {

  class Rotation : public RigidBody {
    protected:
      std::vector<PolygonPoint*> *contour;
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
    public:
      Rotation();
      void setContour(std::vector<PolygonPoint*> *contour_) {
        contour=contour_;
      }

  };

}

#endif
