#ifndef _OPENMBV_SPHERE_H
#define _OPENMBV_SPHERE_H

#include <openmbvcppinterface/rigidbody.h>

namespace OpenMBV {

  class Sphere : public RigidBody {
    protected:
      double radius;
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
    public:
      Sphere();
      void setRadius(double radius_) {
        radius=radius_;
      } 
  };

}

#endif
