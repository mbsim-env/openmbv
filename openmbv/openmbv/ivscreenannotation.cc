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
#include "ivscreenannotation.h"
#include "openmbvcppinterface/ivscreenannotation.h"
#include "mainwindow.h"
#include "utils.h"
#include <Inventor/nodes/SoAlphaTest.h>
#include <Inventor/nodes/SoLineSet.h>

using namespace std;

namespace OpenMBVGUI {

IvScreenAnnotation::IvScreenAnnotation(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : Body(obj, parentItem, soParent, ind) {
  ivsa=std::static_pointer_cast<OpenMBV::IvScreenAnnotation>(obj);
  iconFile="ivscreenannotation.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  auto *pathSep=new SoSeparator;
  pathCoord=new SoCoordinate3;
  pathCoord->point.setNum(0);
  pathSep->addChild(pathCoord);
  pathLine=new SoLineSet;
  pathLine->numVertices.setNum(0);
  pathSep->addChild(pathLine);
  pathMaxFrameRead=-1;

  sep = new SoSeparator;
  if(!clone)
    MainWindow::getInstance()->getScreenAnnotationList()->addChild(sep);
  else
    MainWindow::getInstance()->getScreenAnnotationList()->replaceChild(static_cast<IvScreenAnnotation*>(clone)->sep, sep);

  if(!ivsa->getEnable())
    return;

  if(ivsa->getScale1To1()) {
    auto center = ivsa->getScale1To1At();

    auto trans1 = new SoTranslation;
    trans1->translation.setValue(center[0],center[1],0);
    sep->addChild(trans1);

    sep->addChild(MainWindow::getInstance()->getScreenAnnotationScale1To1());

    auto trans3 = new SoTranslation;
    trans3->translation.setValue(-center[0],-center[1],0);
    sep->addChild(trans3);
  }

  auto *columnLabelFieldSep = new SoSeparator;
  columnLabelFields.resize(ivsa->getColumnLabels().size());
  sep->addChild(columnLabelFieldSep);
  for(size_t i=0; i<ivsa->getColumnLabels().size(); ++i) {
    columnLabelFields[i] = new SoAlphaTest;
    columnLabelFieldSep->addChild(columnLabelFields[i]);
    columnLabelFields[i]->setName(ivsa->getColumnLabels()[i].c_str());
  }

  string fileName=ivsa->getIvFileName();
  SoSeparator *ivSep;
  if(!fileName.empty())
    ivSep = Utils::SoDBreadAllFileNameCached(fileName);
  else
    ivSep = Utils::SoDBreadAllContentCached(ivsa->getIvContent());
  sep->addChild(ivSep);

  auto *pathNode=SoNode::getByName("OpenMBVIvScreenAnnotationPathOrigin");
  if(pathNode) {
    auto *pathSepInIv=dynamic_cast<SoSeparator*>(SoNode::getByName("OpenMBVIvScreenAnnotationPathSep"));
    if(!pathSepInIv)
      throw runtime_error("The node named OpenMBVIvScreenAnnotationPathSep must exist and be of type Separator!");
    pathSepInIv->addChild(pathSep);

    sa = make_unique<SoSearchAction>();
    sa->setNode(pathNode);
    sa->apply(ivSep);
    pathPath = sa->getPath();
    gma = make_unique<SoGetMatrixAction>(SbViewportRegion());
  }
}

IvScreenAnnotation::~IvScreenAnnotation() {
  MainWindow::getInstance()->getScreenAnnotationList()->removeChild(sep);
}

double IvScreenAnnotation::update() {
  if(ivsa->getRows()==0) return 0; // do nothing for environement objects

  // read from hdf5
  int frame=MainWindow::getInstance()->getFrame()->getValue();
  vector<double> data=ivsa->getRow(frame);
  
  auto setColumnLabelFields = [this](const vector<double> &data) {
    for(size_t i=0; i<columnLabelFields.size(); ++i)
      columnLabelFields[i]->value.setValue(data[i]);
  };
  setColumnLabelFields(data);

  // path
  if(pathPath) {
    for(int i=pathMaxFrameRead+1; i<=frame; i++) {
      vector<double> data=ivsa->getRow(i);
      setColumnLabelFields(data);

      gma->setViewportRegion(MainWindow::getInstance()->glViewer->getViewportRegion());
      gma->apply(pathPath);
      SbVec3f translation;
      SbRotation rotation;
      SbVec3f scalevector;
      SbRotation scaleorientation;
      gma->getMatrix().getTransform(translation, rotation, scalevector, scaleorientation);
      pathCoord->point.set1Value(i, translation.getValue());
    }
    pathMaxFrameRead=frame;
    pathLine->numVertices.setValue(1+frame);
  }

  return std::numeric_limits<double>::quiet_NaN();
}

}
