/* OpenMBV - Open Multi Body Viewer.
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
#include "spineextrusion.h"
#include "mainwindow.h"
#include <Inventor/nodes/SoMaterial.h>
#include "utils.h"
#include "openmbvcppinterface/spineextrusion.h"
#include <cfloat>

using namespace std;

namespace OpenMBVGUI {

// copied from SoVRMLExtrusion for including twist in case of collinear spine points 
static float my_normalize(SbVec3f & vec) {
  float len = vec.length();
  if (len > FLT_EPSILON) {
    vec /= len;
  }
  return len;
}

static SbVec3f calculate_y_axis(const SbVec3f * spine, const int i, const int numspine, const SbBool closed)
{
  SbVec3f Y;
  if (closed) {
    if (i > 0) {
      Y = spine[i+1] - spine[i-1];
    }
    else {
      // use numspine-2, since for closed spines, the last spine point == the first point
      Y = spine[1] - spine[numspine >= 2 ? numspine-2 : numspine-1];
    }
  }
  else {
    if (i == 0) Y = spine[1] - spine[0];
    else if (i == numspine-1) Y = spine[numspine-1] - spine[numspine-2];
    else Y = spine[i+1] - spine[i-1];
  }
  my_normalize(Y);
  return Y;
}

static SbVec3f calculate_z_axis(const SbVec3f * spine, const int i, const int numspine, const SbBool closed) {
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
    if (numspine == 2) return {0.0f, 0.0f, 0.0f};
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

SpineExtrusion::SpineExtrusion(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) :
  DynamicColoredBody(obj, parentItem, soParent, ind),
  numberOfSpinePoints(0),
  twistAxis(0.,1.,0.),
  collinear(true), additionalTwist(0.)
{
  spineExtrusion=std::static_pointer_cast<OpenMBV::SpineExtrusion>(obj);

  std::vector<double> data;

  if( spineExtrusion->getStateOffSet().size() > 0 ) {
    data = std::vector<double>(spineExtrusion->getStateOffSet().size()+1);

    for( size_t i = 0; i < spineExtrusion->getStateOffSet().size(); ++i )
      data[i+1] = spineExtrusion->getStateOffSet()[i]; // we have == 0.0 due to local init

    //xml dataset
//  numberOfSpinePoints = int((spineExtrusion->getStateOffSet().size())/(3+3));
    numberOfSpinePoints = int((spineExtrusion->getStateOffSet().size())/(3+6));
  } else {
    //h5 dataset
    data = spineExtrusion->getRow(0);
//  numberOfSpinePoints = int((spineExtrusion->getRow(1).size()-1)/(3+3));
    numberOfSpinePoints = int((spineExtrusion->getRow(1).size()-1)/(3+6));
  }
  int rows=spineExtrusion->getRows();
  double dt;
  if(rows>=2) dt=spineExtrusion->getRow(1)[0]-spineExtrusion->getRow(0)[0]; else dt=0;
  resetAnimRange(rows, dt);

  // read XML
  shared_ptr<vector<shared_ptr<OpenMBV::PolygonPoint> > > contour=spineExtrusion->getContour();

  // create so

  // body
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
  for(int i=0;i<numberOfSpinePoints;i++) {
//  data_coin[i] = SbVec3f(data[(3+3)*i+1],data[(3+3)*i+2],data[(3+3)*i+3]);
    data_coin[i] = SbVec3f(data[(3+6)*i+1],data[(3+6)*i+2],data[(3+6)*i+3]);
  }

  for(int i=0;i<numberOfSpinePoints;i++) {
      SbVec3f Z = calculate_z_axis(data_coin.data(), i, numberOfSpinePoints, false);
      if(Z!=empty)
        collinear=false;
  }

  if(collinear || true) {
    auto *rotation = new SoRotation; // set rotation matrix 
    rotation->ref();
    std::vector<double> rotation_parameter = spineExtrusion->getInitialRotation();
    rotation->rotation.setValue(Utils::cardan2Rotation(SbVec3f(rotation_parameter[0],rotation_parameter[1],rotation_parameter[2]))); 
    SbMatrix Orientation;
    rotation->rotation.getValue().getValue(Orientation);
    additionalTwist = acos(Orientation[2][2]);
    std::cout << "additionalTwist = " << additionalTwist << "\n";

    rotation->unref();
  }

  if( spineExtrusion->getStateOffSet().size() > 0 ) {
    setIvSpine(data);
  }

}

void SpineExtrusion::createProperties() {
  DynamicColoredBody::createProperties();

  if(!clone) {
    properties->updateHeader();
    // GUI editors
    FloatMatrixEditor *contourEditor=new FloatMatrixEditor(properties, QIcon(), "Contour", 0, 3);
    contourEditor->setOpenMBVParameter(spineExtrusion, &OpenMBV::SpineExtrusion::getContour, &OpenMBV::SpineExtrusion::setContour);

    FloatEditor *scaleFactorEditor=new FloatEditor(properties, QIcon(), "Scale factor");
    scaleFactorEditor->setRange(0, DBL_MAX);
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

  setIvSpine(data);

  return data[0];
}

namespace
{
  template<typename OT>
  OT& operator<<(OT& os_, const SbVec3f& v)
  {
    os_ << "[" << setw(12) << v[0]
               << setw(12) << v[1]
               << setw(12) << v[2] << "]";
    return os_;
  }
}

void SpineExtrusion::setIvSpine(const std::vector<double>& data) {
  // set spine
  extrusion->spine.setNum(numberOfSpinePoints);
  SbVec3f *sp = extrusion->spine.startEditing();
  for(int i=0;i<numberOfSpinePoints;i++) {
//  sp[i] = SbVec3f(data[(3+3)*i+1],data[(3+3)*i+2],data[(3+3)*i+3]);
    sp[i] = SbVec3f(data[(3+6)*i+1],data[(3+6)*i+2],data[(3+6)*i+3]);
  }
  extrusion->spine.finishEditing();
  extrusion->spine.setDefault(FALSE);

  if( rotate )
  {
    extrusion->orientation.setNum(numberOfSpinePoints);
    SbRotation *sr = extrusion->orientation.startEditing();

  SbBool colinear = FALSE;
  // test if spine point are collinear
  const SbVec3f empty(0.0f, 0.0f, 0.0f);
  SbVec3f prevY(0.0f, 0.0f, 0.0f);
  SbVec3f prevZ(0.0f, 0.0f, 0.0f);

  SbVec3f X, Y, Z;

  // find first non-collinear spine segments and calculate the first
  // valid Y and Z axis
  for (int i = 0; i < numberOfSpinePoints && (prevY == empty || prevZ == empty); i++) {
    if (prevY == empty) {
      Y = calculate_y_axis(sp, i, numberOfSpinePoints, false);
      if (Y != empty) prevY = Y;
    }
    if (prevZ == empty) {
      Z = calculate_z_axis(sp, i, numberOfSpinePoints, false);
      if (Z != empty) prevZ = Z;
    }
  }

  if (prevY == empty) prevY = SbVec3f(0.0f, 1.0f, 0.0f);
  if (prevZ == empty) { // all spine segments are colinear, calculate constant Z axis
    prevZ = SbVec3f(0.0f, 0.0f, 1.0f);
    if (prevY != SbVec3f(0.0f, 1.0f, 0.0f)) {
      SbRotation rot(SbVec3f(0.0f, 1.0f, 0.0f), prevY);
      rot.multVec(prevZ, prevZ);
    }
    colinear = TRUE;
  }

  // loop through all spines
  for (int i = 0; i < numberOfSpinePoints; i++) {
    Y = calculate_y_axis(sp, i, numberOfSpinePoints, false);
    if (colinear) {
      Z = prevZ;
    }
    else {
      Z = calculate_z_axis(sp, i, numberOfSpinePoints, false);
      if (Z == empty) Z = prevZ;
      if (Z.dot(prevZ) < 0) Z = -Z;
    }

    X = Y.cross(Z);
    my_normalize(X);

    prevZ = Z;

    SbMatrix matrix;
    matrix[0][0] = X[0];
    matrix[0][1] = X[1];
    matrix[0][2] = X[2];
    matrix[0][3] = 0.0f;

    matrix[1][0] = Y[0];
    matrix[1][1] = Y[1];
    matrix[1][2] = Y[2];
    matrix[1][3] = 0.0f;

    matrix[2][0] = Z[0];
    matrix[2][1] = Z[1];
    matrix[2][2] = Z[2];
    matrix[2][3] = 0.0f;

    matrix[3][0] = 0.0f;
    matrix[3][1] = 0.0f;
    matrix[3][2] = 0.0f;
    matrix[3][3] = 1.0f;


    SbRotation I = SbRotation( matrix ).inverse();

#if 1
    Y = SbVec3f( data[(3+6)*i + 4],  data[(3+6)*i + 5],  data[(3+6)*i + 6]);
    Z = SbVec3f( data[(3+6)*i + 7],  data[(3+6)*i + 8],  data[(3+6)*i + 9]);
    X = Y.cross(Z);
    my_normalize(X);

    matrix[0][0] = X[0];
    matrix[0][1] = X[1];
    matrix[0][2] = X[2];
    matrix[0][3] = 0.0f;

    matrix[1][0] = Y[0];
    matrix[1][1] = Y[1];
    matrix[1][2] = Y[2];
    matrix[1][3] = 0.0f;

    matrix[2][0] = Z[0];
    matrix[2][1] = Z[1];
    matrix[2][2] = Z[2];
    matrix[2][3] = 0.0f;

    matrix[3][0] = 0.0f;
    matrix[3][1] = 0.0f;
    matrix[3][2] = 0.0f;
    matrix[3][3] = 1.0f;
#else

    SbRotation T = Utils::cardan2Rotation( SbVec3f(
         data[6*i+4],
         data[6*i+5],
         data[6*i+6]
          ) );
#endif

    sr[i] = SbRotation( matrix ) * I;

    if( i% 5 == 0 )
    {
      std::cout << "   " << data[(3+6)*i+4] << "   " << data[(3+6)*i+5] << "   " << data[(3+6)*i+6] << std::endl;
      std::cout << "   " << data[(3+6)*i+7] << "   " << data[(3+6)*i+8] << "   " << data[(3+6)*i+9] << std::endl;
    }
  }



    extrusion->orientation.finishEditing();
    extrusion->orientation.setDefault(FALSE);
  }
  else
  {
    // set twist
    extrusion->orientation.setNum(numberOfSpinePoints);
    SbRotation *tw = extrusion->orientation.startEditing();
    for(int i=0;i<numberOfSpinePoints;i++) {
      tw[i] = SbRotation(twistAxis,data[(3+6)*i+6]+additionalTwist);
    }
    extrusion->orientation.finishEditing();
    extrusion->orientation.setDefault(FALSE);
  }
}

}
