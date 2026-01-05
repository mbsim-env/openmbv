#include <config.h>
#include "SoDecomposeArray1fToVec3fEngine.h"
#include <Inventor/fields/SoSFVec3f.h>

namespace OpenMBVGUI {

SO_ENGINE_SOURCE(DecomposeArray1fToVec3fEngine);

void DecomposeArray1fToVec3fEngine::initClass() {
  SO_ENGINE_INIT_CLASS(DecomposeArray1fToVec3fEngine, SoEngine, "Engine");
}

DecomposeArray1fToVec3fEngine::DecomposeArray1fToVec3fEngine() {
  SO_ENGINE_CONSTRUCTOR(DecomposeArray1fToVec3fEngine);
  SO_ENGINE_ADD_INPUT(startIndex, (0));
  SO_ENGINE_ADD_INPUT(input, (0));
  SO_ENGINE_ADD_OUTPUT(output, SoSFVec3f);
}

void DecomposeArray1fToVec3fEngine::evaluate() {
  int idx = startIndex.getValue();
  float x = input[idx+0];
  float y = input[idx+1];
  float z = input[idx+2];
  SO_ENGINE_OUTPUT(output, SoSFVec3f, setValue(x,y,z));
}

}
