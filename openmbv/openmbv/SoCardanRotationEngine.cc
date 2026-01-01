#include <config.h>
#include "SoCardanRotationEngine.h"
#include <Inventor/fields/SoSFRotation.h>
#include <Inventor/SbMatrix.h>
#include <openmbv/utils.h>

namespace OpenMBVGUI {

SO_ENGINE_SOURCE(CardanRotationEngine);

void CardanRotationEngine::initClass() {
  SO_ENGINE_INIT_CLASS(CardanRotationEngine, SoEngine, "Engine");
}

CardanRotationEngine::CardanRotationEngine() {
  SO_ENGINE_CONSTRUCTOR(CardanRotationEngine);
  SO_ENGINE_ADD_INPUT(angle, (0,0,0));
  SO_ENGINE_ADD_INPUT(inverse, (false));
  SO_ENGINE_ADD_OUTPUT(rotation, SoSFRotation);
}

void CardanRotationEngine::evaluate() {
  auto T = Utils::cardan2Rotation(angle.getValue());
  if(inverse.getValue())
    T.invert();
  SO_ENGINE_OUTPUT(rotation, SoSFRotation, setValue(T));
}

}
