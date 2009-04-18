#ifndef _AMVIS_CUBE_H_
#define _AMVIS_CUBE_H_

#include <amviscppinterface/rigidbody.h>

namespace AMVis {

  class Cube : public RigidBody {
    protected:
      double length;
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
    public:
      Cube();
      void setLength(double length_) {
        length=length_;
      } 
  };

}

#endif
