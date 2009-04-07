#ifndef _BODY_H_
#define _BODY_H_

#include "object.h"
#include "tinyxml.h"
#include <Inventor/sensors/SoFieldSensor.h>
#include <H5Cpp.h>

class Body : public Object {
  Q_OBJECT
  protected:
    virtual void update()=0;
    SoSwitch *soOutLineSwitch;
    SoSeparator *soOutLineSep;
    QAction *outLine;
  public:
    Body(TiXmlElement* element, H5::Group *h5Parent);
    static SoSFUInt32 *frame;
    static void frameSensorCB(void *data, SoSensor*);
    virtual QMenu* createMenu();
  public slots:
    void outLineSlot();
};

#endif
