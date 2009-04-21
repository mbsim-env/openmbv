#ifndef _OPENMBV_INVISIBLEBODY_H_
#define _OPENMBV_INVISIBLEBODY_H_

#include <openmbvcppinterface/rigidbody.h>

namespace OpenMBV {

  class InvisibleBody : public RigidBody {
    protected:
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
    public:
      InvisibleBody();
  };

}

#endif