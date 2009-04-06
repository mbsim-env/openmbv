#ifndef _OBJECT_H_
#define _OBJECT_H_

#include <QtGui/QTreeWidgetItem>
#include <string>
#include <vector>
#include "tinyxml.h"
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoSwitch.h>
#include <Inventor/fields/SoSFUInt32.h>

#define MBVISNS "{http://www.amm.mw.tum.de/AMVis}"

class Object : public QObject, public QTreeWidgetItem {
  Q_OBJECT
  protected:
    SoSwitch *soSwitch;
    SoSeparator *soSep;
    QAction *draw;
    bool drawThisPath;
    static SoSFUInt32 *frame;
  public:
    Object(TiXmlElement* element);
    SoSwitch* getSoSwitch() { return soSwitch; }
    static std::vector<double> toVector(std::string str); // convenience
    static SoSeparator* soFrame(double size, double offset); // convenience
    virtual QMenu* createMenu();
    void setEnableRecursive(bool enable);
  public slots:
    void drawSlot();
};

#endif
