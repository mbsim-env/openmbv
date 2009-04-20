#ifndef _ARROW_H_
#define _ARROW_H_

#include "config.h"
#include "body.h"
#include "tinyxml.h"
#include <Inventor/nodes/SoBaseColor.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoLineSet.h>
#include <Inventor/nodes/SoScale.h>
#include <Inventor/nodes/SoRotation.h>
#include <QtGui/QMenu>
#include <H5Cpp.h>
#include <hdf5serie/vectorserie.h>

class Arrow : public Body {
  Q_OBJECT
  protected:
    enum Type {
        line,
        fromHead,
        toHead,
        bothHeads
      };
    Type type;
    QAction *path;
    SoSwitch *soPathSwitch;
    SoCoordinate3 *pathCoord;
    SoLineSet *pathLine;
    SoBaseColor *color;
    SoTranslation *toPoint;
    SoRotation *rotation1, *rotation2;
    int pathMaxFrameRead;
    virtual double update();
    H5::VectorSerie<double> *h5Data;
    SoScale *scale1, *scale2;
    double headLength;
    std::vector<double> data;
    double length, scaleLength;
  public:
    Arrow(TiXmlElement* element, H5::Group *h5Parent);
    virtual QString getInfo();
    QMenu* createMenu();
  public slots:
    void pathSlot();
};

#endif
