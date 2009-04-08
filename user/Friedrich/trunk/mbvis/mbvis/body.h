#ifndef _BODY_H_
#define _BODY_H_

#include "object.h"
#include "tinyxml.h"
#include <Inventor/sensors/SoFieldSensor.h>
#include <H5Cpp.h>
#include <QtGui/QActionGroup>
#include <Inventor/nodes/SoDrawStyle.h>

class Body : public Object {
  Q_OBJECT
  private:
    enum DrawStyle { filled, lines, points };
    SoDrawStyle *drawStyle;
  protected:
    virtual void update()=0;
    SoSwitch *soOutLineSwitch;
    SoSeparator *soOutLineSep;
    QAction *outLine;
    QActionGroup *drawMethod;
    QAction *drawMethodPolygon, *drawMethodLine, *drawMethodPoint;
  protected slots:
    void drawMethodSlot(QAction* action);
  public:
    Body(TiXmlElement* element, H5::Group *h5Parent);
    static SoSFUInt32 *frame;
    static void frameSensorCB(void *data, SoSensor*);
    virtual QMenu* createMenu();
  public slots:
    void outLineSlot();
};

#endif
