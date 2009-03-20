#ifndef _AMVIS_CUBOID_H_
#define _AMVIS_CUBOID_H_

#include <amviscppinterface/rigidbody.h>

namespace AMVis {

  class Cuboid : public RigidBody {
    protected:
      std::vector<double> length;
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
    public:
      Cuboid();
      void setLength(const std::vector<double>& length_) {
        assert(length_.size()==3);
        length=length_;
      } 
  };

}

#endif
