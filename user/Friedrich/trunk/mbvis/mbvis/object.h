#ifndef _OBJECT_H_
#define _OBJECT_H_

#include <QtGui/QTreeWidgetItem>
#include <string>
#include <vector>
#include "tinyxml.h"
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoSwitch.h>
#include <Inventor/fields/SoSFUInt32.h>
#include <H5Cpp.h>
#include <map>

#define MBVISNS "{http://www.amm.mw.tum.de/AMVis}"

class Object : public QObject, public QTreeWidgetItem {
  Q_OBJECT
  protected:
    SoSwitch *soSwitch;
    SoSeparator *soSep;
    QAction *draw;
    bool drawThisPath;
    H5::Group *h5Group;
  public:
    Object(TiXmlElement* element, H5::Group *h5Parent);
    SoSwitch* getSoSwitch() { return soSwitch; }
    static std::vector<double> toVector(std::string str); // convenience
    static SoSeparator* soFrame(double size, double offset); // convenience
    virtual QMenu* createMenu();
    void setEnableRecursive(bool enable);
    static std::map<SoNode*,Object*> objectMap;
    std::string getPath();
    virtual QString getInfo();
    std::string iconFile;
  public slots:
    void drawSlot();
};

#endif
