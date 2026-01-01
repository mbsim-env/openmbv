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

#ifndef _OPENMBVGUI_VIOBJECT_H_
#define _OPENMBVGUI_VIOBJECT_H_

#include "body.h"

namespace OpenMBV {
  class DynamicIvBody;
}

class SoShaderParameterArray1f;
class SoShaderParameter1f;

namespace OpenMBVGUI {

class DynamicIvBody : public Body {
  Q_OBJECT
  public:
    DynamicIvBody(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind);
    ~DynamicIvBody() override;
  protected:
    std::shared_ptr<OpenMBV::DynamicIvBody> ivo;
    double update() override;
    SoShaderParameterArray1f *dataNodeVector;
    std::vector<SoShaderParameter1f*> dataNodeScalar;
    std::vector<double> data;
    std::vector<float> datamfmf;
};

}

#endif
