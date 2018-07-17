/*
    OpenMBV - Open Multi Body Viewer.
    Copyright (C) 2009 Markus Friedrich

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include "config.h"
#include "objectfactory.h"

#include "openmbvcppinterface/arrow.h"
#include "openmbvcppinterface/coilspring.h"
#include "openmbvcppinterface/compoundrigidbody.h"
#include "openmbvcppinterface/cube.h"
#include "openmbvcppinterface/cuboid.h"
#include "openmbvcppinterface/extrusion.h"
#include "openmbvcppinterface/rotation.h"
#include "openmbvcppinterface/frame.h"
#include "openmbvcppinterface/grid.h"
#include "openmbvcppinterface/frustum.h"
#include "openmbvcppinterface/group.h"
#include "openmbvcppinterface/invisiblebody.h"
#include "openmbvcppinterface/ivbody.h"
#include "openmbvcppinterface/nurbsdisk.h"
#include "openmbvcppinterface/nurbscurve.h"
#include "openmbvcppinterface/nurbssurface.h"
#include "openmbvcppinterface/dynamicnurbscurve.h"
#include "openmbvcppinterface/dynamicnurbssurface.h"
#include "openmbvcppinterface/pointset.h"
#include "openmbvcppinterface/indexedlineset.h"
#include "openmbvcppinterface/indexedfaceset.h"
#include "openmbvcppinterface/dynamicpointset.h"
#include "openmbvcppinterface/dynamicindexedlineset.h"
#include "openmbvcppinterface/dynamicindexedfaceset.h"
#include "openmbvcppinterface/path.h"
#include "openmbvcppinterface/sphere.h"
#include "openmbvcppinterface/spineextrusion.h"

#include "arrow.h"
#include "coilspring.h"
#include "compoundrigidbody.h"
#include "cube.h"
#include "cuboid.h"
#include "extrusion.h"
#include "rotation.h"
#include "frame.h"
#include "grid.h"
#include "frustum.h"
#include "group.h"
#include "invisiblebody.h"
#include "iostream"
#include "ivbody.h"
#include "mainwindow.h"
#include "nurbsdisk.h"
#include "nurbscurve.h"
#include "nurbssurface.h"
#include "dynamicnurbscurve.h"
#include "dynamicnurbssurface.h"
#include "pointset.h"
#include "indexedlineset.h"
#include "indexedfaceset.h"
#include "dynamicpointset.h"
#include "dynamicindexedlineset.h"
#include "dynamicindexedfaceset.h"
#include "path.h"
#include "sphere.h"
#include "spineextrusion.h"
#include <string>

#if BOOST_VERSION >= 105600
  #include <boost/core/demangle.hpp>
#else
  #include <cxxabi.h>
  #ifndef BOOST_CORE_DEMANGLE_REPLACEMENT
  #define BOOST_CORE_DEMANGLE_REPLACEMENT
  namespace boost {
    namespace core {
      inline std::string demangle(const std::string &name) {
        int status;
        char* retc=abi::__cxa_demangle(name.c_str(), nullptr, nullptr, &status);
        if(status!=0) throw std::runtime_error("Cannot demangle c++ symbol.");
        std::string ret(retc);
        free(retc);
        return ret;
      }
    }
  }
  #endif
#endif

using namespace std;

namespace OpenMBVGUI {

Object *ObjectFactory::create(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) {
  if(typeid(*obj)==typeid(OpenMBV::Group))
    return new Group(obj, parentItem, soParent, ind);
  else if(typeid(*obj)==typeid(OpenMBV::Arrow))
    return new Arrow(obj, parentItem, soParent, ind);
  else if(typeid(*obj)==typeid(OpenMBV::CoilSpring))
    return new CoilSpring(obj, parentItem, soParent, ind);
  else if(typeid(*obj)==typeid(OpenMBV::CompoundRigidBody))
    return new CompoundRigidBody(obj, parentItem, soParent, ind);
  else if(typeid(*obj)==typeid(OpenMBV::Cube))
    return new Cube(obj, parentItem, soParent, ind);
  else if(typeid(*obj)==typeid(OpenMBV::Cuboid))
    return new Cuboid(obj, parentItem, soParent, ind);
  else if(typeid(*obj)==typeid(OpenMBV::Extrusion))
    return new Extrusion(obj, parentItem, soParent, ind);
  else if(typeid(*obj)==typeid(OpenMBV::Rotation))
    return new Rotation(obj, parentItem, soParent, ind);
  else if(typeid(*obj)==typeid(OpenMBV::Grid))
    return new Grid(obj, parentItem, soParent, ind);
  else if(typeid(*obj)==typeid(OpenMBV::Frame))
    return new Frame(obj, parentItem, soParent, ind);
  else if(typeid(*obj)==typeid(OpenMBV::Frustum))
    return new Frustum(obj, parentItem, soParent, ind);
  else if(typeid(*obj)==typeid(OpenMBV::IvBody))
    return new IvBody(obj, parentItem, soParent, ind);
  else if(typeid(*obj)==typeid(OpenMBV::InvisibleBody))
    return new InvisibleBody(obj, parentItem, soParent, ind);
  else if(typeid(*obj)==typeid(OpenMBV::NurbsDisk))
    return new NurbsDisk(obj, parentItem, soParent, ind);
  else if(typeid(*obj)==typeid(OpenMBV::NurbsCurve))
    return new NurbsCurve(obj, parentItem, soParent, ind);
  else if(typeid(*obj)==typeid(OpenMBV::NurbsSurface))
    return new NurbsSurface(obj, parentItem, soParent, ind);
  else if(typeid(*obj)==typeid(OpenMBV::DynamicNurbsCurve))
    return new DynamicNurbsCurve(obj, parentItem, soParent, ind);
  else if(typeid(*obj)==typeid(OpenMBV::DynamicNurbsSurface))
    return new DynamicNurbsSurface(obj, parentItem, soParent, ind);
  else if(typeid(*obj)==typeid(OpenMBV::PointSet))
    return new PointSet(obj, parentItem, soParent, ind);
  else if(typeid(*obj)==typeid(OpenMBV::IndexedLineSet))
    return new IndexedLineSet(obj, parentItem, soParent, ind);
  else if(typeid(*obj)==typeid(OpenMBV::IndexedFaceSet))
    return new IndexedFaceSet(obj, parentItem, soParent, ind);
  else if(typeid(*obj)==typeid(OpenMBV::DynamicPointSet))
    return new DynamicPointSet(obj, parentItem, soParent, ind);
  else if(typeid(*obj)==typeid(OpenMBV::DynamicIndexedLineSet))
    return new DynamicIndexedLineSet(obj, parentItem, soParent, ind);
  else if(typeid(*obj)==typeid(OpenMBV::DynamicIndexedFaceSet))
    return new DynamicIndexedFaceSet(obj, parentItem, soParent, ind);
  else if(typeid(*obj)==typeid(OpenMBV::Path))
    return new Path(obj, parentItem, soParent, ind);
  else if(typeid(*obj)==typeid(OpenMBV::Sphere))
    return new Sphere(obj, parentItem, soParent, ind);
  else if(typeid(*obj)==typeid(OpenMBV::SpineExtrusion))
    return new SpineExtrusion(obj, parentItem, soParent, ind);
  QString str("Unknown OpenMBV::Object: %1"); str=str.arg(boost::core::demangle(typeid(*obj).name()).c_str());
  MainWindow::getInstance()->statusBar()->showMessage(str, 10000);
  msgStatic(Warn)<<str.toStdString()<<endl;
  return nullptr;
}


}
