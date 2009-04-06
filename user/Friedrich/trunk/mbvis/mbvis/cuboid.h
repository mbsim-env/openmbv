#ifndef _CUBOID_H_
#define _CUBOID_H_

#include "rigidbody.h"
#include <string>
#include "tinyxml.h"

class Cuboid : public RigidBody {
  Q_OBJECT
  public:
    Cuboid(TiXmlElement* element);
};

#endif
