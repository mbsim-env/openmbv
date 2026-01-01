#include <config.h>
#include "SoStringFormatEngine.h"
#include <Inventor/fields/SoSFRotation.h>
#include <Inventor/SbMatrix.h>
#include <openmbv/utils.h>
#include <boost/format.hpp>
#include <boost/algorithm/string/replace.hpp>


namespace OpenMBVGUI {

SO_ENGINE_SOURCE(StringFormatEngine);

void StringFormatEngine::initClass() {
  SO_ENGINE_INIT_CLASS(StringFormatEngine, SoEngine, "Engine");
}

StringFormatEngine::StringFormatEngine() {
  SO_ENGINE_CONSTRUCTOR(StringFormatEngine);
  SO_ENGINE_ADD_INPUT(i0, (0));
  SO_ENGINE_ADD_INPUT(i1, (0));
  SO_ENGINE_ADD_INPUT(i2, (0));
  SO_ENGINE_ADD_INPUT(i3, (0));
  SO_ENGINE_ADD_INPUT(i4, (0));
  SO_ENGINE_ADD_INPUT(i5, (0));
  SO_ENGINE_ADD_INPUT(i6, (0));
  SO_ENGINE_ADD_INPUT(i7, (0));
  SO_ENGINE_ADD_INPUT(i8, (0));
  SO_ENGINE_ADD_INPUT(i9, (0));
  SO_ENGINE_ADD_INPUT(f0, (0));
  SO_ENGINE_ADD_INPUT(f1, (0));
  SO_ENGINE_ADD_INPUT(f2, (0));
  SO_ENGINE_ADD_INPUT(f3, (0));
  SO_ENGINE_ADD_INPUT(f4, (0));
  SO_ENGINE_ADD_INPUT(f5, (0));
  SO_ENGINE_ADD_INPUT(f6, (0));
  SO_ENGINE_ADD_INPUT(f7, (0));
  SO_ENGINE_ADD_INPUT(f8, (0));
  SO_ENGINE_ADD_INPUT(f9, (0));
  SO_ENGINE_ADD_INPUT(s0, (""));
  SO_ENGINE_ADD_INPUT(s1, (""));
  SO_ENGINE_ADD_INPUT(s2, (""));
  SO_ENGINE_ADD_INPUT(s3, (""));
  SO_ENGINE_ADD_INPUT(s4, (""));
  SO_ENGINE_ADD_INPUT(s5, (""));
  SO_ENGINE_ADD_INPUT(s6, (""));
  SO_ENGINE_ADD_INPUT(s7, (""));
  SO_ENGINE_ADD_INPUT(s8, (""));
  SO_ENGINE_ADD_INPUT(s9, (""));
  SO_ENGINE_ADD_INPUT(b0, (0));
  SO_ENGINE_ADD_INPUT(b1, (0));
  SO_ENGINE_ADD_INPUT(b2, (0));
  SO_ENGINE_ADD_INPUT(b3, (0));
  SO_ENGINE_ADD_INPUT(b4, (0));
  SO_ENGINE_ADD_INPUT(b5, (0));
  SO_ENGINE_ADD_INPUT(b6, (0));
  SO_ENGINE_ADD_INPUT(b7, (0));
  SO_ENGINE_ADD_INPUT(b8, (0));
  SO_ENGINE_ADD_INPUT(b9, (0));
  SO_ENGINE_ADD_INPUT(format, (""));
  SO_ENGINE_ADD_OUTPUT(output, SoSFString);
}

void StringFormatEngine::evaluate() {
  std::string formatStr = format.getValue().getString();
  if(formatStr != currentFormat) {
    static const std::vector<std::string> keys {
      "i0", "i1", "i2", "i3", "i4", "i5", "i6", "i7", "i8", "i9",
      "f0", "f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9",
      "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9",
      "b0", "b1", "b2", "b3", "b4", "b5", "b6", "b7", "b8", "b9",
    };

    convertedFormat = formatStr+"%"+std::to_string(keys.size()+1)+"%";
    for(size_t i=0; i<keys.size(); ++i) {
      boost::algorithm::replace_all(convertedFormat, "%"+keys[i]+"%" , "%"+std::to_string(i+1)+"%");
      boost::algorithm::replace_all(convertedFormat, "%"+keys[i]+"$" , "%"+std::to_string(i+1)+"$");
      boost::algorithm::replace_all(convertedFormat, "%|"+keys[i]+"$", "%|"+std::to_string(i+1)+"$");
    }
  }

  #define V(x) x.getValue()
  #define S(x) x.getValue().getString()
  auto outStr = (boost::format(convertedFormat) %
    V(i0) % V(i1) % V(i2) % V(i3) % V(i4) % V(i5) % V(i6) % V(i7) % V(i8) % V(i9) %
    V(f0) % V(f1) % V(f2) % V(f3) % V(f4) % V(f5) % V(f6) % V(f7) % V(f8) % V(f9) %
    S(s0) % S(s1) % S(s2) % S(s3) % S(s4) % S(s5) % S(s6) % S(s7) % S(s8) % S(s9) %
    V(b0) % V(b1) % V(b2) % V(b3) % V(b4) % V(b5) % V(b6) % V(b7) % V(b8) % V(b9) %
    "").str();
  #undef V
  #undef S
  SO_ENGINE_OUTPUT(output, SoSFString, setValue(outStr.data()));
}

}
