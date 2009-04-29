#ifndef _SOSEPNOPICKNOBBOX_H_
#define _SOSEPNOPICKNOBBOX_H_

#include "config.h"
#include <Inventor/nodes/SoSeparator.h>

class SoSepNoPickNoBBox : public SoSeparator {
  public:
    void rayPick(SoRayPickAction *action) {}
    void getBoundingBox(SoGetBoundingBoxAction *action) {}
};

#endif
