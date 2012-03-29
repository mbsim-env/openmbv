#include "SoTransposeEngine.h"
#include <Inventor/SbLinear.h>

SO_ENGINE_SOURCE(SoTransposeEngine);

void SoTransposeEngine::initClass() {
  SO_ENGINE_INIT_CLASS(SoTransposeEngine, SoEngine, "Engine");
}

SoTransposeEngine::SoTransposeEngine() {
  SO_ENGINE_CONSTRUCTOR(SoTransposeEngine);
  SO_ENGINE_ADD_INPUT(inRotation, (0,0,0,1));
  SO_ENGINE_ADD_OUTPUT(outRotation, SoSFRotation);
}

void SoTransposeEngine::evaluate() {
  SbMatrix matrix;
  inRotation.getValue().getValue(matrix);
  SO_ENGINE_OUTPUT(outRotation, SoSFRotation, setValue(SbRotation(matrix.transpose())));
}
