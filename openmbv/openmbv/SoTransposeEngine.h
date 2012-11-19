#include <Inventor/engines/SoSubEngine.h>
#include <Inventor/fields/SoSFRotation.h>

namespace OpenMBVGUI {

class SoTransposeEngine : public SoEngine {
 SO_ENGINE_HEADER(SoTransposeEngine);
 public:
   SoSFRotation inRotation; // input
   SoEngineOutput outRotation; // output (SoSFRotation)

   static void initClass();
   SoTransposeEngine();

 private:
   virtual ~SoTransposeEngine() {}
   virtual void evaluate();
};

}
