#ifndef _OPENMBV_FRAME_H
#define _OPENMBV_FRAME_H

#include <openmbvcppinterface/rigidbody.h>

namespace OpenMBV {

  class Frame : public RigidBody {
    protected:
      double size;
      double offset;
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
    public:
      Frame();
      void setSize(double size_) { size=size_; }
      void setOffset(double offset_) { offset=offset_; }
  };

}

#endif
