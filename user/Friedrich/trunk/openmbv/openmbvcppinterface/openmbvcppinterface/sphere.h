#ifndef _OPENMBV_SPHERE_H
#define _OPENMBV_SPHERE_H

#include <openmbvcppinterface/rigidbody.h>

namespace OpenMBV {

  /** A spehere */
  class Sphere : public RigidBody {
    protected:
      double radius;
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
    public:
      /** Default constructor */
      Sphere();

      /** Set the radius of the shpere */
      void setRadius(double radius_) {
        radius=radius_;
      } 
  };

}

#endif
