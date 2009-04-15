#ifndef _INVISIBLEBODY_H_
#define _INVISIBLEBODY_H_

#include "config.h"
#include "rigidbody.h"
#include <string>
#include "tinyxml.h"
#include <H5Cpp.h>

class InvisibleBody : public RigidBody {
  Q_OBJECT
  public:
    InvisibleBody(TiXmlElement* element, H5::Group *h5Parent);
};

#endif
