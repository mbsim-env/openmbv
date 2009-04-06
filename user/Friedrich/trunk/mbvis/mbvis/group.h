#ifndef _GROUP_H_
#define _GROUP_H_

#include "object.h"
#include <string>
#include "tinyxml.h"

class Group : public Object {
  Q_OBJECT
  protected:
    virtual void update() {}
  public:
    Group(TiXmlElement *element);
};

#endif
