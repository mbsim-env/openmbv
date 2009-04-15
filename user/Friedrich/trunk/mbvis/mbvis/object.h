#ifndef _OBJECT_H_
#define _OBJECT_H_

#include "config.h"
#include <QtGui/QTreeWidgetItem>
#include <string>
#include <vector>
#include "tinyxml.h"
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoSwitch.h>
#include <Inventor/nodes/SoCube.h>
#include <Inventor/fields/SoSFUInt32.h>
#include <H5Cpp.h>
#include <map>
#include <Inventor/sensors/SoNodeSensor.h>
#include <Inventor/nodes/SoTranslation.h>

#define MBVISNS "{http://www.amm.mw.tum.de/AMVis}"

class Object : public QObject, public QTreeWidgetItem {
  Q_OBJECT
  protected:
    SoSwitch *soSwitch;
    SoSeparator *soSep;
    QAction *draw;
    bool drawThisPath;
    H5::Group *h5Group;
    SoSwitch *soBBoxSwitch;
    SoSeparator *soBBoxSep;
    SoTranslation *soBBoxTrans;
    SoCube *soBBox;
    QAction *bbox;
  public:
    Object(TiXmlElement* element, H5::Group *h5Parent);
    SoSwitch* getSoSwitch() { return soSwitch; }
    virtual QMenu* createMenu();
    void setEnableRecursive(bool enable);
    static std::map<SoNode*,Object*> objectMap;
    std::string getPath();
    virtual QString getInfo();
    std::string iconFile;
    static void nodeSensorCB(void *data, SoSensor*);
  public slots:
    void drawSlot();
    void bboxSlot();
};

#endif
