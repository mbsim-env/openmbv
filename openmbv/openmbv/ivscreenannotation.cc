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

  // load the IV content using a cache
  string fileName=ivsa->getIvFileName();
  struct SoUnref  { void operator()(SoSeparator *s) { if(s) s->unref(); } };
  unique_ptr<SoSeparator, SoUnref> ivSep;
  if(!fileName.empty())
    ivSep.reset(Utils::SoDBreadAllFileNameCached(fileName));
  else
    ivSep.reset(Utils::SoDBreadAllContentCached(ivsa->getIvContent()));
  if(!ivSep)
    return;
  ivSep->ref();

  // search for a OpenMBVIvScreenAnnotationPathOrigin or OpenMBVIvScreenAnnotationPathOrigin1 node
  auto getPathNode = [](SoSeparator *ivSep) {
    auto *pathNode=Utils::getChildNodeByName(ivSep, "OpenMBVIvScreenAnnotationPathOrigin");
    if(!pathNode)
      pathNode=Utils::getChildNodeByName(ivSep, "OpenMBVIvScreenAnnotationPathOrigin1");
    return pathNode;
  };
  auto pathNode = getPathNode(ivSep.get());
  // if such a node exists we cannot use the cached content -> use a copy and search pathNode again in the copy
  if(pathNode) {
    ivSep.reset(static_cast<SoSeparator*>(ivSep->copy(true)));
    ivSep->ref();
    pathNode = getPathNode(ivSep.get());
  }

  // add the cached IV content or the copied content to the scene graph
  auto ivSepPtr = ivSep.release();
  sep->addChild(ivSepPtr);

  // now search for the pathSep node without the number (only done one, not inside of the loop)
  SoSeparator *pathSepNoNumber = nullptr, *pathSepInIv;
  if(pathNode)
    pathSepNoNumber=dynamic_cast<SoSeparator*>(Utils::getChildNodeByName(ivSepPtr, "OpenMBVIvScreenAnnotationPathSep"));

  // loop over all pathNode's
  int i=1;
  while(pathNode) {
    // if a pathSep node without a number exists, use it, if not search a pathSep node with the corresponding number
    if(pathSepNoNumber)
      pathSepInIv=pathSepNoNumber;
    else
      pathSepInIv=dynamic_cast<SoSeparator*>(Utils::getChildNodeByName(ivSepPtr, ("OpenMBVIvScreenAnnotationPathSep"+to_string(i)).c_str()));
    // stop if not such node was found
    if(!pathSepInIv) {
      cerr<<"A node named OpenMBVIvScreenAnnotationPathSep[<nr>] must exist and be of type Separator!"<<endl;
      break;
    }

    // create the nodes needed for the path
    auto *pathSep=new SoSeparator;
    pathSepInIv->addChild(pathSep);
    pathCoord.emplace_back(new SoCoordinate3);
    pathCoord.back()->point.setNum(0);
    pathSep->addChild(pathCoord.back());
    pathLine.emplace_back(new SoLineSet);
    pathLine.back()->numVertices.setNum(0);
    pathSep->addChild(pathLine.back());

    SoSearchAction sa;
    sa.setNode(pathNode);
    sa.apply(ivSepPtr);
    pathPath.emplace_back(sa.getPath());
    pathPath.back()->ref();
    gma = make_unique<SoGetMatrixAction>(SbViewportRegion());

    i++;
    pathNode=Utils::getChildNodeByName(ivSepPtr, ("OpenMBVIvScreenAnnotationPathOrigin"+to_string(i)).c_str());
  }
  pathMaxFrameRead=-1;
}

IvScreenAnnotation::~IvScreenAnnotation() {
  MainWindow::getInstance()->getScreenAnnotationList()->removeChild(sep);
  for(auto pp : pathPath)
    pp->unref();
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
  for(int i=pathMaxFrameRead+1; i<=frame; i++) {
    vector<double> data=ivsa->getRow(i);
    setColumnLabelFields(data);
    for(size_t idx=0; idx<pathPath.size(); ++idx) {
      gma->setViewportRegion(MainWindow::getInstance()->glViewer->getViewportRegion());
      gma->apply(pathPath[idx]);
      SbVec3f translation;
      SbRotation rotation;
      SbVec3f scalevector;
      SbRotation scaleorientation;
      gma->getMatrix().getTransform(translation, rotation, scalevector, scaleorientation);
      pathCoord[idx]->point.set1Value(i, translation.getValue());
    }
  }
  for(size_t idx=0; idx<pathPath.size(); ++idx)
    pathLine[idx]->numVertices.setValue(1+frame);
  pathMaxFrameRead=frame;

  return std::numeric_limits<double>::quiet_NaN();
}

}
