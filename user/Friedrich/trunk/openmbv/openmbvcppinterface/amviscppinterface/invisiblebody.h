#ifndef _AMVIS_INVISIBLEBODY_H_
#define _AMVIS_INVISIBLEBODY_H_

#include <amviscppinterface/rigidbody.h>

namespace AMVis {

  class InvisibleBody : public RigidBody {
    protected:
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
    public:
      InvisibleBody();
  };

}

#endif
