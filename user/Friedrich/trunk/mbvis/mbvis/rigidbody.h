#ifndef _RIGIDBODY_H_
#define _RIGIDBODY_H_

#include "body.h"
#include "tinyxml.h"
#include <Inventor/nodes/SoSwitch.h>
#include <Inventor/nodes/SoTranslation.h>
#include <Inventor/nodes/SoRotationXYZ.h>
#include <Inventor/nodes/SoBaseColor.h>

class RigidBody : public Body {
  Q_OBJECT
  protected:
    QAction *localFrame, *referenceFrame;
    SoSwitch *soLocalFrameSwitch, *soReferenceFrameSwitch;
    virtual void update();
    SoRotationXYZ *rotationAlpha, *rotationBeta, *rotationGamma;
    SoTranslation *translation;
    SoBaseColor *color;
  public:
    RigidBody(TiXmlElement* element);
    virtual QMenu* createMenu();
  public slots:
    void localFrameSlot();
    void referenceFrameSlot();
};

#endif
