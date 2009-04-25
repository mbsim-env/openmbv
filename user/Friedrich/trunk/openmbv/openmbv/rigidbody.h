#ifndef _RIGIDBODY_H_
#define _RIGIDBODY_H_

#include "config.h"
#include "body.h"
#include "tinyxml.h"
#include <Inventor/nodes/SoSwitch.h>
#include <Inventor/nodes/SoTranslation.h>
#include <Inventor/nodes/SoRotationXYZ.h>
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoLineSet.h>
#include <H5Cpp.h>
#include <hdf5serie/vectorserie.h>

class RigidBody : public Body {
  Q_OBJECT
  protected:
    QAction *localFrame, *referenceFrame, *path;
    SoSwitch *soLocalFrameSwitch, *soReferenceFrameSwitch, *soPathSwitch;
    SoCoordinate3 *pathCoord;
    SoLineSet *pathLine;
    int pathMaxFrameRead;
    virtual double update();
    SoRotationXYZ *rotationAlpha, *rotationBeta, *rotationGamma;
    SoTranslation *translation;
    SoMaterial *mat;
    H5::VectorSerie<double> *h5Data;
    SoScale *refFrameScale, *localFrameScale;
  public:
    RigidBody(TiXmlElement* element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent);
    virtual QMenu* createMenu();
    virtual QString getInfo();
  public slots:
    void localFrameSlot();
    void referenceFrameSlot();
    void pathSlot();
};

#endif
