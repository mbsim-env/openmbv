#ifndef _SPHERE_H_
#define _SPHERE_H_

#include "config.h"
#include "rigidbody.h"
#include <string>
#include "tinyxml.h"
#include <H5Cpp.h>

class Sphere : public RigidBody {
  Q_OBJECT
  public:
    Sphere(TiXmlElement* element, H5::Group *h5Parent);
};

#endif
