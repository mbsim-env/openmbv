#include <config.h>
#include "SoSpecial.h"

namespace OpenMBVGUI {

SO_NODE_SOURCE(SepNoPickNoBBox);

void SepNoPickNoBBox::initClass()
{
  SO_NODE_INIT_CLASS(SepNoPickNoBBox, SoSeparator, "SoSeparator");
}

SepNoPickNoBBox::SepNoPickNoBBox()
{
  SO_NODE_CONSTRUCTOR(SepNoPickNoBBox);
}



SO_NODE_SOURCE(SepNoPick);

void SepNoPick::initClass()
{
  SO_NODE_INIT_CLASS(SepNoPick, SoSeparator, "SoSeparator");
}

SepNoPick::SepNoPick()
{
  SO_NODE_CONSTRUCTOR(SepNoPick);
}



SO_NODE_SOURCE(BaseColorHeavyOverride);

void BaseColorHeavyOverride::initClass()
{
  SO_NODE_INIT_CLASS(BaseColorHeavyOverride, SoBaseColor, "SoBaseColor");
}

BaseColorHeavyOverride::BaseColorHeavyOverride()
{
  SO_NODE_CONSTRUCTOR(BaseColorHeavyOverride);
}

}
