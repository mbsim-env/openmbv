#ifndef _BODY_H_
#define _BODY_H_

#include "config.h"
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
    static bool existFiles;
  public:
    virtual void update()=0;
    SoSwitch *soOutLineSwitch;
    SoSeparator *soOutLineSep;
    QAction *outLine;
    QActionGroup *drawMethod;
    QAction *drawMethodPolygon, *drawMethodLine, *drawMethodPoint;
    void resetAnimRange(int numOfRows, double dt);
  protected slots:
    void drawMethodSlot(QAction* action);
  public:
    Body(TiXmlElement* element, H5::Group *h5Parent);
    static void frameSensorCB(void *data, SoSensor*);
    virtual QMenu* createMenu();
    static std::vector<double> toVector(std::string str); // convenience
    static SoSeparator* soFrame(double size, double offset); // convenience
  public slots:
    void outLineSlot();
};

#endif
