/* OpenMBV - Open Multi Body Viewer.
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
#include "spineextrusion.h"
#include "mainwindow.h"
#include <Inventor/nodes/SoMaterial.h>
#include <Inventor/nodes/SoNormal.h>
#include <Inventor/nodes/SoIndexedTriangleStripSet.h>
#include "utils.h"
#include "openmbvcppinterface/spineextrusion.h"
#include <cfloat>

using namespace std;

namespace OpenMBVGUI {

// BEGIN this code is copied from SoVRMLExtrusion, see below
static float
my_normalize(SbVec3f & vec)
{
  float len = vec.length();
  if (len > FLT_EPSILON) {
    vec /= len;
  }
  return len;
}

static SbVec3f
calculate_z_axis(const SbVec3f * spine, const int i,
                 const int numspine, const SbBool closed)
{
  SbVec3f z0, z1;

  if (closed) {
    if (i > 0) {
      if (i == numspine-1) {
        z0 = spine[1] - spine[i];
        z1 = spine[i-1] - spine[i];
      }
      else {
        z0 = spine[i+1] - spine[i];
        z1 = spine[i-1] - spine[i];
      }
    }
    else {
      z0 = spine[1] - spine[0];
      z1 = spine[numspine >= 2 ? numspine-2 : numspine-1] - spine[0];
    }
  }
  else {
    if (numspine == 2) return SbVec3f(0.0f, 0.0f, 0.0f);
    else if (i == 0) {
      z0 = spine[2] - spine[1];
      z1 = spine[0] - spine[1];
    }
    else if (i == numspine-1) {
      z0 = spine[numspine-1] - spine[numspine-2];
      z1 = spine[numspine-3] - spine[numspine-2];
    }
    else {
      z0 = spine[i+1] - spine[i];
      z1 = spine[i-1] - spine[i];
    }
  }

  my_normalize(z0);
  my_normalize(z1);

  // test if spine segments are collinear. If they are, the cross
  // product will not be reliable, and we should just use the previous
  // Z-axis instead.
  if (SbAbs(z0.dot(z1)) > 0.999f) {
    return SbVec3f(0.0f, 0.0f, 0.0f);
  }
  SbVec3f tmp = z0.cross(z1);
  if (my_normalize(tmp) == 0.0f) {
    return SbVec3f(0.0f, 0.0f, 0.0f);
  }
  return tmp;
}
// END this code is copied from SoVRMLExtrusion

SpineExtrusion::SpineExtrusion(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) :
  DynamicColoredBody(obj, parentItem, soParent, ind),
  numberOfSpinePoints(0),
  twistAxis(0.,1.,0.),
  collinear(true), additionalTwist(0.)
{
  spineExtrusion=std::static_pointer_cast<OpenMBV::SpineExtrusion>(obj);

  switch(spineExtrusion->getCrossSectionOrientation()) {
    case OpenMBV::SpineExtrusion::orthogonalWithTwist: doublesPerPoint = 4; break;
    case OpenMBV::SpineExtrusion::cardanWrtWorld     : doublesPerPoint = 6; break;
  }

  std::vector<double> data;

  if( spineExtrusion->getStateOffSet().size() > 0 ) {
    data = std::vector<double>(spineExtrusion->getStateOffSet().size()+1);

    for( size_t i = 0; i < spineExtrusion->getStateOffSet().size(); ++i )
      data[i+1] = spineExtrusion->getStateOffSet()[i]; // we have == 0.0 due to local init

    //xml dataset
    numberOfSpinePoints = int((spineExtrusion->getStateOffSet().size())/doublesPerPoint);
  } else {
    //h5 dataset
    data = spineExtrusion->getRow(0);
    numberOfSpinePoints = int((spineExtrusion->getRow(1).size()-1)/doublesPerPoint);
  }
  int rows=spineExtrusion->getRows();
  double dt;
  if(rows>=2) dt=spineExtrusion->getRow(1)[0]-spineExtrusion->getRow(0)[0]; else dt=0;
  resetAnimRange(rows, dt);

  // read XML
  shared_ptr<vector<shared_ptr<OpenMBV::PolygonPoint> > > contour=spineExtrusion->getContour();

  // create so

  // body
  switch(spineExtrusion->getCrossSectionOrientation()) {
    case OpenMBV::SpineExtrusion::orthogonalWithTwist: {
      extrusion=new SoVRMLExtrusion;
      soSep->addChild(extrusion);

      // scale
      extrusion->scale.setNum(numberOfSpinePoints);
      SbVec2f *sc = extrusion->scale.startEditing();
      for(int i=0;i<numberOfSpinePoints;i++) sc[i] = SbVec2f(spineExtrusion->getScaleFactor(),spineExtrusion->getScaleFactor()); // first x-scale / second z-scale
      extrusion->scale.finishEditing();
      extrusion->scale.setDefault(FALSE);

      // cross section
      extrusion->crossSection.setNum(contour?contour->size()+1:0);
      SbVec2f *cs = extrusion->crossSection.startEditing();
      for(size_t i=0;i<(contour?contour->size():0);i++) cs[i] = SbVec2f((*contour)[i]->getXComponent(), (*contour)[i]->getYComponent()); // clockwise in local coordinate system
      if(contour) cs[contour->size()] =  SbVec2f((*contour)[0]->getXComponent(), (*contour)[0]->getYComponent()); // closed cross section
      extrusion->crossSection.finishEditing();
      extrusion->crossSection.setDefault(FALSE);

      // additional flags
      extrusion->solid=TRUE; // backface culling
      extrusion->convex=TRUE; // only convex polygons included in visualisation
      extrusion->ccw=TRUE; // vertex ordering counterclockwise?
      extrusion->beginCap=TRUE; // front side at begin of the spine
      extrusion->endCap=TRUE; // front side at end of the spine
      extrusion->creaseAngle=0.3; // angle below which surface normals are drawn smooth

      // test if spine point are collinear
      const SbVec3f empty(0.0f, 0.0f, 0.0f);

      std::vector<SbVec3f> data_coin(numberOfSpinePoints);
      for(int i=0;i<numberOfSpinePoints;i++)
        data_coin[i] = SbVec3f(data[doublesPerPoint*i+1],data[doublesPerPoint*i+2],data[doublesPerPoint*i+3]);

      for(int i=0;i<numberOfSpinePoints;i++) {
          SbVec3f Z = calculate_z_axis(data_coin.data(), i, numberOfSpinePoints, false);
          if(Z!=empty)
            collinear=false;
      }

      if(collinear) {
        auto *rotation = new SoRotation; // set rotation matrix 
        rotation->ref();
        std::vector<double> rotation_parameter = spineExtrusion->getInitialRotation();
        rotation->rotation.setValue(Utils::cardan2Rotation(SbVec3f(rotation_parameter[0],rotation_parameter[1],rotation_parameter[2]))); 
        SbMatrix Orientation;
        rotation->rotation.getValue().getValue(Orientation);
        additionalTwist = acos(Orientation[2][2]);
        rotation->unref();
      }

      if( spineExtrusion->getStateOffSet().size() > 0 ) {
        setIvSpine(data);
      }
      break;
    }
    case OpenMBV::SpineExtrusion::cardanWrtWorld: {
      quadMeshCoords = new SoCoordinate3;
      soSep->addChild(quadMeshCoords);
      quadMeshCoords->point.setNum(numberOfSpinePoints * contour->size());
      quadMeshNormals = new SoNormal;
      soSep->addChild(quadMeshNormals);
      quadMeshNormals->vector.setNum(numberOfSpinePoints * 2*contour->size());
      auto stripMesh = new SoIndexedTriangleStripSet;
      soSep->addChild(stripMesh);

      // mesh indices of spine
      stripMesh->coordIndex.setNum((2*numberOfSpinePoints+1) * contour->size());
      int *p = stripMesh->coordIndex.startEditing();
      int idx=0;
      for(size_t csIdx=0;csIdx<contour->size();csIdx++) {
        for(int spIdx=0;spIdx<numberOfSpinePoints;spIdx++) {
          int coordIdx = spIdx*contour->size()+csIdx;
          p[idx++] = coordIdx;
          p[idx++] = csIdx<contour->size()-1 ? coordIdx+1 : coordIdx+1-contour->size();
        }
        p[idx++] = -1;
      }
      stripMesh->coordIndex.finishEditing();

      // normal indices of spine
      stripMesh->normalIndex.setNum((2*numberOfSpinePoints+1) * contour->size());
      int *n = stripMesh->normalIndex.startEditing();
      idx=0;
      for(size_t csIdx=0;csIdx<contour->size();csIdx++) {
        for(int spIdx=0;spIdx<numberOfSpinePoints;spIdx++) {
          int coordIdx = spIdx*contour->size()+csIdx;
          n[idx++] = 2*coordIdx;
          n[idx++] = 2*coordIdx+1;
        }
        n[idx++] = -1;
      }
      stripMesh->normalIndex.finishEditing();

      int csSize = contour->size();

      // end cups as tesselation
      {
        auto endCupSep = new SoSeparator;
        soSep->addChild(endCupSep);
        // normal binding
        auto *nb=new SoNormalBinding;
        endCupSep->addChild(nb);
        nb->value.setValue(SoNormalBinding::OVERALL);
        // normal
        auto *endCupNormal=new SoNormal;
        endCupSep->addChild(endCupNormal);
        endCupNormal->vector.set1Value(0, 0,0,1);
        // coords
        auto endCupPoint=new SoCoordinate3;
        endCupSep->addChild(endCupPoint);
        endCupPoint->point.setNum(csSize);
        auto p = endCupPoint->point.startEditing();
        idx=0;
        for(size_t csIdx=0; csIdx<contour->size(); csIdx++)
          p[idx++] = SbVec3f(
            (*contour)[csIdx]->getXComponent() * spineExtrusion->getScaleFactor(),
            0,
            (*contour)[csIdx]->getYComponent() * spineExtrusion->getScaleFactor()
          );
        endCupPoint->point.finishEditing();
        // tesselation
        auto endCup=new IndexedTesselationFace;
        endCup->windingRule.setValue(IndexedTesselationFace::ODD);
        endCup->coordinate.connectFrom(&endCupPoint->point);
        endCup->coordIndex.setNum(csSize+2);
        auto *ec = endCup->coordIndex.startEditing();
        idx=0;
        for(size_t csIdx=0; csIdx<contour->size(); csIdx++) {
          ec[idx] = idx;
          idx++;
        }
        ec[idx++] = 0;
        ec[idx++] = -1;
        endCup->coordIndex.finishEditing();
        endCup->generate();

        for(int i : {0,1}) {
          auto sep = new SoSeparator;
          endCupSep->addChild(sep);
          // vertex ordering
          auto *sh=new SoShapeHints;
          sep->addChild(sh);
          sh->vertexOrdering.setValue(i==0 ? SoShapeHints::CLOCKWISE : SoShapeHints::COUNTERCLOCKWISE);
          sh->shapeType.setValue(SoShapeHints::SOLID);
          // translation/rotation and face
          endCupTrans[i] = new SoTranslation;
          sep->addChild(endCupTrans[i]);
          endCupRot[i] = new SoRotation;
          sep->addChild(endCupRot[i]);
          sep->addChild(endCup);
        }
      }

      // outline
      soSep->addChild(soOutLineSwitch);

      // outline indices of spine
      auto *ol=new SoIndexedLineSet;
      soOutLineSep->addChild(quadMeshCoords);
      soOutLineSep->addChild(ol);
      int csSize1 = std::count_if(contour->begin(), contour->end(), [](const auto &c) { return round(c->getBorderValue())==1; });
      ol->coordIndex.setNum((numberOfSpinePoints+1)*csSize1);
      auto *l = ol->coordIndex.startEditing();
      idx = 0;
      for(size_t csIdx=0;csIdx<contour->size();csIdx++) {
        if(round((*contour)[csIdx]->getBorderValue())==0)
          continue;
        for(int spIdx=0;spIdx<numberOfSpinePoints;spIdx++) {
          int pIdx = spIdx*csSize+csIdx;
          l[idx++] = pIdx;
        }
        l[idx++] = -1;
      }
      ol->coordIndex.finishEditing();

      // outline indices of end cups
      idx = ol->coordIndex.getNum();
      ol->coordIndex.setNum(idx+2*csSize+4);
      l = ol->coordIndex.startEditing();
      for(int spIdx : {0, numberOfSpinePoints-1}) {
        for(size_t csIdx=0;csIdx<contour->size();csIdx++) {
          int pIdx = spIdx*csSize+csIdx;
          l[idx++] = pIdx;
        }
        l[idx++] = spIdx*csSize;
        l[idx++] = -1;
      }
      ol->coordIndex.finishEditing();

      setCardanWrtWorldSpine(data);
      break;
    }
  }

}

void SpineExtrusion::createProperties() {
  DynamicColoredBody::createProperties();

  if(!clone) {
    properties->updateHeader();
    // GUI editors
    auto *contourEditor=new FloatMatrixEditor(properties, QIcon(), "Contour", 0, 3);
    contourEditor->setOpenMBVParameter(spineExtrusion, &OpenMBV::SpineExtrusion::getContour, &OpenMBV::SpineExtrusion::setContour);

    auto *scaleFactorEditor=new FloatEditor(properties, QIcon(), "Scale factor");
    scaleFactorEditor->setOpenMBVParameter(spineExtrusion, &OpenMBV::SpineExtrusion::getScaleFactor, &OpenMBV::SpineExtrusion::setScaleFactor);
  }
}

QString SpineExtrusion::getInfo() {
  return DynamicColoredBody::getInfo();
}

double SpineExtrusion::update() {
  if(spineExtrusion->getRows()==0) return 0; // do nothing for environement objects

  // read from hdf5
  int frame=MainWindow::getInstance()->getFrame()->getValue();
  std::vector<double> data=spineExtrusion->getRow(frame);

  if( spineExtrusion->getStateOffSet().size() > 0 )
    for( size_t i = 0; i < spineExtrusion->getStateOffSet().size(); ++i )
      data[i+1] += spineExtrusion->getStateOffSet()[i];

  switch(spineExtrusion->getCrossSectionOrientation()) {
    case OpenMBV::SpineExtrusion::orthogonalWithTwist:
      setIvSpine(data);
      break;
    case OpenMBV::SpineExtrusion::cardanWrtWorld:
      setCardanWrtWorldSpine(data);
      break;
  }

  return data[0];
}

void SpineExtrusion::setIvSpine(const std::vector<double>& data) {
  // set spine
  extrusion->spine.setNum(numberOfSpinePoints);
  SbVec3f *sp = extrusion->spine.startEditing();
  for(int i=0;i<numberOfSpinePoints;i++)
    sp[i] = SbVec3f(data[doublesPerPoint*i+1],data[doublesPerPoint*i+2],data[doublesPerPoint*i+3]);
  extrusion->spine.finishEditing();
  extrusion->spine.setDefault(FALSE);

  // set orientation
  extrusion->orientation.setNum(numberOfSpinePoints);
  SbRotation *tw = extrusion->orientation.startEditing();
  // The twist angle (data[4*i+4]) from HDF5 is around the local y-axis (twistAxis).
  // And for a initially collinear curve a constant additionalTwist is added.
  // (to detect a initially collinear curve we use exactly the same code as in Coin, see above)
  for(int i=0;i<numberOfSpinePoints;i++)
    tw[i] = SbRotation(twistAxis,data[4*i+4]+additionalTwist);
  extrusion->orientation.finishEditing();
  extrusion->orientation.setDefault(FALSE);
}

void SpineExtrusion::setCardanWrtWorldSpine(const std::vector<double> &data) {
  auto &contour = *spineExtrusion->getContour();
  int csSize = contour.size();

  SbVec3f r01[2];
  SbMatrix T01[2];

  // points
  {
    SbVec3f *p = quadMeshCoords->point.startEditing();
    for(int spIdx=0; spIdx<numberOfSpinePoints; spIdx++) {
      SbVec3f r(data[spIdx*6+1],data[spIdx*6+2],data[spIdx*6+3]);
      SbVec3f angle(data[spIdx*6+4],data[spIdx*6+5],data[spIdx*6+6]);
      SbMatrix T;
      Utils::cardan2Rotation(angle).getValue(T);
      if(spIdx==0) {
        r01[0] = r;
        T01[0] = T;
      }
      if(spIdx==numberOfSpinePoints-1) {
        r01[1] = r;
        T01[1] = T;
      }
      for(int csIdx=0; csIdx<csSize; csIdx++) {
        SbVec3f nsp(
          contour[csIdx]->getXComponent() * spineExtrusion->getScaleFactor(),
          0,
          contour[csIdx]->getYComponent() * spineExtrusion->getScaleFactor()
        );
        SbVec3f T_nsp;
        T.multMatrixVec(nsp, T_nsp);
        int pIdx = spIdx*csSize+csIdx;
        p[pIdx] = r + T_nsp;
      }
    }
    quadMeshCoords->point.finishEditing();
  }

  // normals
  {
    auto p = quadMeshCoords->point.getValues(0);
    SbVec3f *n = quadMeshNormals->vector.startEditing();
    // normals: smooth in spine direction and face-based in cross-section direction
    for(int spIdx=0; spIdx<numberOfSpinePoints; spIdx++) {
      for(int csIdx=0; csIdx<csSize; csIdx++) {
        int pIdx = spIdx*csSize+csIdx;
        int nIdx = spIdx*2*csSize+2*csIdx;
        {
          int bspS=pIdx+(spIdx==0?+0:-csSize);
          int bspE=pIdx+(spIdx==numberOfSpinePoints-1?-0:csSize);
          SbVec3f bsp = p[bspE] - p[bspS];
          SbVec3f bcs = p[csIdx==csSize-1?pIdx+1-csSize:pIdx+1] - p[pIdx];
          n[nIdx] = bcs.cross(bsp);
          n[nIdx].normalize();
        }
        {
          int bspS=pIdx+1+(spIdx==0?+0:-csSize);
          int bspE=pIdx+1+(spIdx==numberOfSpinePoints-1?-0:csSize);
          if(csIdx==csSize-1) { bspS-=csSize; bspE-=csSize; }
          SbVec3f bsp = p[bspE] - p[bspS];
          SbVec3f bcs = p[csIdx==csSize-1?pIdx+1-csSize:pIdx+1] - p[pIdx+1-1];
          n[nIdx+1] = bcs.cross(bsp);
          n[nIdx+1].normalize();
        }
      }
    }
    // normals: combine (make smooth) the normals of ajected faces if getBorderValue of the contour is 1
    for(int csIdx=0; csIdx<csSize; csIdx++) {
      if(round(contour[csIdx]->getBorderValue())==1)
        continue;
      for(int spIdx=0; spIdx<numberOfSpinePoints; spIdx++) {
        int nIdx = spIdx*2*csSize+2*csIdx;
        auto &n1 = n[nIdx];
        auto &n2 = n[nIdx+(csIdx==0?2*csSize-1:-1)];
        n1 = n1 + n2;
        n1.normalize();
        n2 = n1;
      }
    }
    quadMeshNormals->vector.finishEditing();
  }

  // translation and rotation of end cups
  for(int i : {0,1}) {
    SbVec3f dummy1, dummy2;
    SbRotation dummy3;
    SbRotation rot;
    T01[i].transpose().getTransform(dummy1, rot, dummy2, dummy3);
    endCupTrans[i]->translation.setValue(r01[i]);
    endCupRot[i]->rotation.setValue(rot);
  }
}

}
