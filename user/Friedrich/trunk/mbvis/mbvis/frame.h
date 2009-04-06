#ifndef _FRAME_H_
#define _FRAME_H_

#include "rigidbody.h"
#include "tinyxml.h"

class Frame : public RigidBody {
  Q_OBJECT
  public:
    Frame(TiXmlElement* element);
};

#endif
