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
      extrusion->ccw=spineExtrusion->getCounterClockWise() ? TRUE : FALSE; // vertex ordering
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
      // outline (DynamicColoredBody does not set soOutLineSwitch as a child of soSep)
      soSep->addChild(soOutLineSwitch);

      extrusionCardan.init(numberOfSpinePoints, contour,
                           spineExtrusion->getScaleFactor(), spineExtrusion->getCounterClockWise(),
                           soSep, soOutLineSep);
      extrusionCardan.setCardanWrtWorldSpine(data);

#if 0
      // draw normals for debugging
      auto normCol = new SoMaterial;
      soSep->addChild(normCol);
      normCol->diffuseColor.setValue(0,0,0);
      normCol->emissiveColor.setValue(0,1,0);
      auto normC = new SoCoordinate3;
      soSep->addChild(normC);
      normC->point.setNum(numberOfSpinePoints * csSize * 2 * 2);
      float k=0.2/5;
      idx=0;
      for(int spIdx=0; spIdx<numberOfSpinePoints; ++spIdx) {
        for(int csIdx=0; csIdx<csSize; ++csIdx) {
          int pIdx = spIdx*csSize+csIdx;
          int nIdx = 2*pIdx;
          normC->point.set1Value(idx++, quadMeshCoords->point[pIdx]);
          normC->point.set1Value(idx++, quadMeshCoords->point[pIdx]+k*quadMeshNormals->vector[csIdx>0?nIdx-1:nIdx-1+2*csSize]);
          normC->point.set1Value(idx++, quadMeshCoords->point[pIdx]);
          normC->point.set1Value(idx++, quadMeshCoords->point[pIdx]+k*quadMeshNormals->vector[nIdx]);
        }
      }
      auto normL = new SoLineSet;
      soSep->addChild(normL);
      normL->numVertices.setNum(numberOfSpinePoints * csSize * 2);
      for(int i=0; i<numberOfSpinePoints * csSize * 2; ++i)
        normL->numVertices.set1Value(i, 2);
#endif

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
  int frame=MainWindow::getInstance()->getFrame()[0];
  std::vector<double> data=spineExtrusion->getRow(frame);

  if( spineExtrusion->getStateOffSet().size() > 0 )
    for( size_t i = 0; i < spineExtrusion->getStateOffSet().size(); ++i )
      data[i+1] += spineExtrusion->getStateOffSet()[i];

  switch(spineExtrusion->getCrossSectionOrientation()) {
    case OpenMBV::SpineExtrusion::orthogonalWithTwist:
      setIvSpine(data);
      break;
    case OpenMBV::SpineExtrusion::cardanWrtWorld:
      extrusionCardan.setCardanWrtWorldSpine(data, spineExtrusion->getUpdateNormals());
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

void ExtrusionCardan::setCardanWrtWorldSpine(const std::vector<double> &data, bool updateNormals) {
  int csSize = nsp.size();
  int spSize = quadMeshCoords->point.getNum() / csSize;

  SbVec3f r01[2];
  SbMatrix T01[2];

  {
    SbVec3f *p = quadMeshCoords->point.startEditing();
    SbVec3f *n = nullptr;
    if(updateNormals)
      n = quadMeshNormals->vector.startEditing();
    //#pragma omp parallel for default(none) shared(data, r01, T01, csSize, p, n, updateNormals)
    for(int spIdx=0; spIdx<spSize; spIdx++) {
      SbVec3f r(data[spIdx*6+1],data[spIdx*6+2],data[spIdx*6+3]);
      SbVec3f angle(data[spIdx*6+4],data[spIdx*6+5],data[spIdx*6+6]);
      SbMatrix T;
      Utils::cardan2Rotation(angle).getValue(T);
      if(spIdx==0) {
        r01[0] = r;
        T01[0] = T;
      }
      if(spIdx==spSize-1) {
        r01[1] = r;
        T01[1] = T;
      }
      for(int csIdx=0; csIdx<csSize; csIdx++) {
        // points
        SbVec3f T_nsp;
        T.multMatrixVec(nsp[csIdx], T_nsp);
        int pIdx = spIdx*csSize+csIdx;
        p[pIdx] = r + T_nsp;
        if(updateNormals) {
          // normals
          int nIdx = spIdx*2*csSize+2*csIdx;
          SbVec3f T_normal;
          T.multMatrixVec(normal[2*csIdx+0], T_normal);
          n[nIdx]   = T_normal;
          T.multMatrixVec(normal[2*csIdx+1], T_normal);
          n[nIdx+1] = T_normal;
        }
      }
    }
    if(updateNormals) {
      //#pragma omp parallel for default(none) shared(csSize, p, n)
      for(int spIdx=0; spIdx<spSize; spIdx++) {
        for(int csIdx=0; csIdx<csSize; csIdx++) {
          // normals x b
          int pIdx = spIdx*csSize+csIdx;
          int nIdx = 2*pIdx;
          int pIdxA = (spIdx>0?spIdx-1:0)*csSize+csIdx;
          int pIdxB = (spIdx<spSize-1?spIdx+1:spSize-1)*csSize+csIdx;
          SbVec3f b = p[pIdxB] - p[pIdxA];
          auto ortho = [](const SbVec3f &n, const SbVec3f &b) {
            auto x = n.dot(b)/b.dot(b)*b;
            return n-x;
          };
          n[nIdx]   = ortho(n[nIdx  ], b);
          n[nIdx+1] = ortho(n[nIdx+1], b);
          n[nIdx]  .normalize();
          n[nIdx+1].normalize();
        }
      }
      quadMeshNormals->vector.finishEditing();
    }
    quadMeshCoords->point.finishEditing();
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

void ExtrusionCardan::init(int spSize, const std::shared_ptr<std::vector<std::shared_ptr<OpenMBV::PolygonPoint> > > &contour,
                           double csScale, bool ccw,
                           SoSeparator *soSep, SoSeparator *soOutLineSep) {
      int csSize = contour->size();

      quadMeshCoords = new SoCoordinate3;
      soSep->addChild(quadMeshCoords);
      quadMeshCoords->point.setNum(spSize * csSize);
      quadMeshNormals = new SoNormal;
      soSep->addChild(quadMeshNormals);
      quadMeshNormals->vector.setNum(spSize * 2*csSize);
      auto *sh=new SoShapeHints;
      soSep->addChild(sh);
      sh->vertexOrdering.setValue(ccw ? SoShapeHints::COUNTERCLOCKWISE : SoShapeHints::CLOCKWISE);
      sh->shapeType.setValue(SoShapeHints::SOLID);
      auto stripMesh = new SoIndexedTriangleStripSet;
      soSep->addChild(stripMesh);

      // mesh indices of spine
      stripMesh->coordIndex.setNum((2*spSize+1) * csSize);
      int *p = stripMesh->coordIndex.startEditing();
      int idx=0;
      for(int csIdx=0;csIdx<csSize;csIdx++) {
        for(int spIdx=0;spIdx<spSize;spIdx++) {
          int coordIdx = spIdx*csSize+csIdx;
          p[idx++] = coordIdx;
          p[idx++] = csIdx<csSize-1 ? coordIdx+1 : coordIdx+1-csSize;
        }
        p[idx++] = -1;
      }
      stripMesh->coordIndex.finishEditing();

      // normal indices of spine
      stripMesh->normalIndex.setNum((2*spSize+1) * csSize);
      int *n = stripMesh->normalIndex.startEditing();
      idx=0;
      for(int csIdx=0;csIdx<csSize;csIdx++) {
        for(int spIdx=0;spIdx<spSize;spIdx++) {
          int coordIdx = spIdx*csSize+csIdx;
          n[idx++] = 2*coordIdx;
          n[idx++] = 2*coordIdx+1;
        }
        n[idx++] = -1;
      }
      stripMesh->normalIndex.finishEditing();

      // end cups as tesselation
      {
        auto endCupSep = new SoSeparator;
        soSep->addChild(endCupSep);
        // normal binding
        auto *nb=new SoNormalBinding;
        endCupSep->addChild(nb);
        nb->value.setValue(SoNormalBinding::OVERALL);
        // coords
        auto endCupPoint=new SoCoordinate3;
        endCupSep->addChild(endCupPoint);
        endCupPoint->point.setNum(csSize);
        auto p = endCupPoint->point.startEditing();
        idx=0;
        for(int csIdx=0; csIdx<csSize; csIdx++)
          p[idx++] = SbVec3f(
            (*contour)[csIdx]->getXComponent() * csScale,
            0,
            (*contour)[csIdx]->getYComponent() * csScale
          );
        endCupPoint->point.finishEditing();
        // tesselation
        auto endCup=new IndexedTesselationFace;
        endCup->windingRule.setValue(IndexedTesselationFace::ODD);
        endCup->coordinate.connectFrom(&endCupPoint->point);
        endCup->coordIndex.setNum(csSize+2);
        auto *ec = endCup->coordIndex.startEditing();
        idx=0;
        for(int csIdx=0; csIdx<csSize; csIdx++) {
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
          // normal
          auto *endCupNormal=new SoNormal;
          sep->addChild(endCupNormal);
          endCupNormal->vector.set1Value(0, 0,!((i==0) xor ccw)?-1:1,0);
          // vertex ordering
          auto *sh=new SoShapeHints;
          sep->addChild(sh);
          sh->vertexOrdering.setValue(!((i==0) xor ccw) ? SoShapeHints::CLOCKWISE : SoShapeHints::COUNTERCLOCKWISE);
          sh->shapeType.setValue(SoShapeHints::SOLID);
          // translation/rotation and face
          endCupTrans[i] = new SoTranslation;
          sep->addChild(endCupTrans[i]);
          endCupRot[i] = new SoRotation;
          sep->addChild(endCupRot[i]);
          sep->addChild(endCup);
        }
      }

      // outline indices of spine
      auto *ol=new SoIndexedLineSet;
      soOutLineSep->addChild(quadMeshCoords);
      soOutLineSep->addChild(ol);
      int csSize1 = std::count_if(contour->begin(), contour->end(), [](const auto &c) { return round(c->getBorderValue())==1; });
      ol->coordIndex.setNum((spSize+1)*csSize1);
      auto *l = ol->coordIndex.startEditing();
      idx = 0;
      for(int csIdx=0;csIdx<csSize;csIdx++) {
        if(round((*contour)[csIdx]->getBorderValue())==0)
          continue;
        for(int spIdx=0;spIdx<spSize;spIdx++) {
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
      for(int spIdx : {0, spSize-1}) {
        for(int csIdx=0;csIdx<csSize;csIdx++) {
          int pIdx = spIdx*csSize+csIdx;
          l[idx++] = pIdx;
        }
        l[idx++] = spIdx*csSize;
        l[idx++] = -1;
      }
      ol->coordIndex.finishEditing();
      
      // prepare contour points (convert to SbVec3f) and calculate normal in cross-section plane
      nsp.resize(csSize);
      normal.resize(2*csSize);
      for(int csIdx=0;csIdx<csSize;csIdx++) {
        // points
        nsp[csIdx].setValue(
          (*contour)[csIdx]->getXComponent() * csScale,
          0,
          (*contour)[csIdx]->getYComponent() * csScale
        );
        // normals
        int nIdx = 2*csIdx;
        normal[nIdx].setValue(
          -((*contour)[csIdx==csSize-1?0:csIdx+1]->getYComponent() - (*contour)[csIdx]->getYComponent()),
          0,
            (*contour)[csIdx==csSize-1?0:csIdx+1]->getXComponent() - (*contour)[csIdx]->getXComponent()
        );
        normal[nIdx].normalize();
        normal[nIdx+1] = normal[nIdx];
      }
      for(int csIdx=0;csIdx<csSize;csIdx++) {
        // combine normals
        int nIdx = 2*csIdx;
        if(round((*contour)[csIdx]->getBorderValue())==0) {
          auto &n1 = normal[csIdx>0?nIdx-1:nIdx-1+2*csSize];
          auto &n2 = normal[nIdx];
          n1 = n1 + n2;
          n1.normalize();
          n2 = n1;
        }
      }

}

}
