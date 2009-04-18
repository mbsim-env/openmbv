#ifndef _FRAME_H_
#define _FRAME_H_

#include "config.h"
#include "rigidbody.h"
#include "tinyxml.h"
#include <H5Cpp.h>

class Frame : public RigidBody {
  Q_OBJECT
  public:
    Frame(TiXmlElement* element, H5::Group *h5Parent);
};

#endif
