#ifndef _CUBE_H_
#define _CUBE_H_

#include "config.h"
#include "rigidbody.h"
#include <string>
#include "tinyxml.h"
#include <H5Cpp.h>

class Cube : public RigidBody {
  Q_OBJECT
  public:
    Cube(TiXmlElement* element, H5::Group *h5Parent);
};

#endif
