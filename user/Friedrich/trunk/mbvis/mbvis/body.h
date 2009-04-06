#ifndef _BODY_H_
#define _BODY_H_

#include "object.h"
#include "tinyxml.h"
#include <Inventor/sensors/SoFieldSensor.h>

class Body : public Object {
  Q_OBJECT
  protected:
    virtual void update()=0;
  public:
    Body(TiXmlElement* element);
    static void frameSensorCB(void *data, SoSensor*);
};

#endif
