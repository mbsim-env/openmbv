/*
    OpenMBV - Open Multi Body Viewer.
    Copyright (C) 2009 Markus Friedrich

  This library is free software; you can redistribute it and/or 
  modify it under the terms of the GNU Lesser General Public 
  License as published by the Free Software Foundation; either 
  version 2.1 of the License, or (at your option) any later version. 
   
  This library is distributed in the hope that it will be useful, 
  but WITHOUT ANY WARRANTY; without even the implied warranty of 
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU 
  Lesser General Public License for more details. 
   
  You should have received a copy of the GNU Lesser General Public 
  License along with this library; if not, write to the Free Software 
  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301 USA
*/

#ifndef _OPENMBVGUI_SOSTRINGFORMATENGINE_H_
#define _OPENMBVGUI_SOSTRINGFORMATENGINE_H_

#include <Inventor/C/errors/debugerror.h> // workaround a include order bug in Coin-3.1.3
#include <Inventor/engines/SoSubEngine.h>
#include <Inventor/fields/SoSFInt32.h>
#include <Inventor/fields/SoSFFloat.h>
#include <Inventor/fields/SoSFString.h>
#include <Inventor/fields/SoSFBool.h>
#include <string>

namespace OpenMBVGUI {

/* String formatting engine.
 *
 * 'format' is a boost-format-library string format specification.
 * However, only the positional syntax variants are supported (%1%, %1$..., %|1$...)
 * and instead of the positional number (1, 2, 3, ...) the name of a input of
 * this engine must be used (e.g. %i0%, %f2$.3f, ...)
 *                                 ^^----^^---input-field-names
 *
 * The following single valued inputs are available:
 * - int32 inputs:  i0, i1, i2, i3, i4, i5, i6, i7, i8, i9
 * - float inputs:  f0, f1, f2, f3, f4, f5, f6, f7, f8, f9
 * - string inputs: s0, s1, s2, s3, s4, s5, s6, s7, s8, s9
 * - bool inputs:   b0, b1, b2, b3, b4, b5, b6, b7, b8, b9
 *
 * 'format' is then substituted with these input (if referenced by 'format')
 * and the resulting string is available as single valued string 'output'.
 */
class StringFormatEngine : public SoEngine {
  SO_ENGINE_HEADER(StringFormatEngine);
  public:
    SoSFInt32   i0, i1, i2, i3, i4, i5, i6, i7, i8, i9;
    SoSFFloat   f0, f1, f2, f3, f4, f5, f6, f7, f8, f9;
    SoSFString  s0, s1, s2, s3, s4, s5, s6, s7, s8, s9;
    SoSFBool    b0, b1, b2, b3, b4, b5, b6, b7, b8, b9;
    SoSFString  format;
    SoEngineOutput output;
 
    static void initClass();
    StringFormatEngine();
 
  private:
    ~StringFormatEngine() override = default;
    void evaluate() override;

    std::string currentFormat;
    std::string convertedFormat;
};

}

#endif
