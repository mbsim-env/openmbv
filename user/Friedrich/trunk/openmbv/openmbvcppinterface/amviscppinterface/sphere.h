#ifndef _AMVIS_SPHERE_H
#define _AMVIS_SPHERE_H

#include <amviscppinterface/rigidbody.h>

namespace AMVis {

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
