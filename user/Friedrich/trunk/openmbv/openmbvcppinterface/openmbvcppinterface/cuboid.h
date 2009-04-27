#ifndef _OPENMBV_CUBOID_H_
#define _OPENMBV_CUBOID_H_

#include <openmbvcppinterface/rigidbody.h>

namespace OpenMBV {

  /** A cuboid */
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
      void setLength(double x, double y, double z) {
        std::vector<double> length_;
        length_.push_back(x);
        length_.push_back(y);
        length_.push_back(z);
        length=length_;
      } 
  };

}

#endif
