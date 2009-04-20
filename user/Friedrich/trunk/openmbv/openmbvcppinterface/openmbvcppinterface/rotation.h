#ifndef _OPENMBV_ROTATION_H_
#define _OPENMBV_ROTATION_H_

#include <openmbvcppinterface/rigidbody.h>
#include <openmbvcppinterface/polygonpoint.h>
#include <vector>

namespace OpenMBV {

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
