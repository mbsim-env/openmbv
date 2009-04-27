#ifndef _OPENMBV_IVBODY_H_
#define _OPENMBV_IVBODY_H_

#include <openmbvcppinterface/rigidbody.h>
#include <string>

namespace OpenMBV {

  /** A body defines by a Open Inventor file or a VRML file */
  class IvBody : public RigidBody {
    public:
      IvBody();
      void setIvFileName(std::string ivFileName_) { ivFileName=ivFileName_; }
    protected:
      std::string ivFileName;
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
  };

}

#endif
