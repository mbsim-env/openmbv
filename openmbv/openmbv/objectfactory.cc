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
#include "openmbvcppinterface/cylindricalgear.h"
#include "openmbvcppinterface/cylinder.h"
#include "openmbvcppinterface/rack.h"
#include "openmbvcppinterface/bevelgear.h"
#include "openmbvcppinterface/planargear.h"

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
#include "cylindricalgear.h"
#include "cylinder.h"
#include "rack.h"
#include "bevelgear.h"
#include "planargear.h"
#include <string>
#include <boost/core/demangle.hpp>

using namespace std;

namespace OpenMBVGUI {

Object *ObjectFactory::create(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) {
  auto &objRef=*obj;
  if(typeid(objRef)==typeid(OpenMBV::Group))
    return new Group(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::Arrow))
    return new Arrow(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::CoilSpring))
    return new CoilSpring(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::CompoundRigidBody))
    return new CompoundRigidBody(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::Cube))
    return new Cube(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::Cuboid))
    return new Cuboid(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::Extrusion))
    return new Extrusion(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::Rotation))
    return new Rotation(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::Grid))
    return new Grid(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::Frame))
    return new Frame(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::Frustum))
    return new Frustum(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::IvBody))
    return new IvBody(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::InvisibleBody))
    return new InvisibleBody(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::NurbsDisk))
    return new NurbsDisk(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::NurbsCurve))
    return new NurbsCurve(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::NurbsSurface))
    return new NurbsSurface(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::DynamicNurbsCurve))
    return new DynamicNurbsCurve(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::DynamicNurbsSurface))
    return new DynamicNurbsSurface(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::PointSet))
    return new PointSet(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::IndexedLineSet))
    return new IndexedLineSet(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::IndexedFaceSet))
    return new IndexedFaceSet(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::DynamicPointSet))
    return new DynamicPointSet(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::DynamicIndexedLineSet))
    return new DynamicIndexedLineSet(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::DynamicIndexedFaceSet))
    return new DynamicIndexedFaceSet(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::Path))
    return new Path(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::Sphere))
    return new Sphere(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::SpineExtrusion))
    return new SpineExtrusion(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::CylindricalGear))
    return new CylindricalGear(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::Cylinder))
    return new Cylinder(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::Rack))
    return new Rack(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::BevelGear))
    return new BevelGear(obj, parentItem, soParent, ind);
  else if(typeid(objRef)==typeid(OpenMBV::PlanarGear))
    return new PlanarGear(obj, parentItem, soParent, ind);
  QString str("Unknown OpenMBV::Object: %1"); str=str.arg(boost::core::demangle(typeid(objRef).name()).c_str());
  MainWindow::getInstance()->statusBar()->showMessage(str, 10000);
  msgStatic(Warn)<<str.toStdString()<<endl;
  return nullptr;
}


}
