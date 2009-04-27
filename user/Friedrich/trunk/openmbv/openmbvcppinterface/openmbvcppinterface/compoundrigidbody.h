#ifndef _OPENMBV_COMPOUNDRIGIDBODY_H_
#define _OPENMBV_COMPOUNDRIGIDBODY_H_

#include <openmbvcppinterface/rigidbody.h>
#include <vector>

namespace OpenMBV {

  /** A compound of rigid bodies */
  class CompoundRigidBody : public RigidBody {
    protected:
      std::vector<RigidBody*> rigidBody;
      void writeXMLFile(std::ofstream& xmlFile, const std::string& indent="");
    public:
      CompoundRigidBody();
      void addRigidBody(RigidBody* rigidBody_) {
        rigidBody.push_back(rigidBody_);
      }
  };

}

#endif
