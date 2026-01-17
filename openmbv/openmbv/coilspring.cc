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
#include "coilspring.h"
#include "spineextrusion.h"
#include "mainwindow.h"
#include "utils.h"
#include "openmbvcppinterface/coilspring.h"
#include <Inventor/nodes/SoLineSet.h>
#include <Inventor/nodes/SoTransform.h>
#include <Inventor/nodes/SoNormal.h>
#include <Inventor/nodes/SoVertexAttribute.h>
#include <QMenu>
#include <boost/dll/runtime_symbol_info.hpp>
#include <cfloat>
#include <QMessageBox>

using namespace std;

namespace OpenMBVGUI {

CoilSpring::CoilSpring(const std::shared_ptr<OpenMBV::Object> &obj, QTreeWidgetItem *parentItem, SoGroup *soParent, int ind) : DynamicColoredBody(obj, parentItem, soParent, ind), scaledSpine(nullptr) {
  coilSpring=std::static_pointer_cast<OpenMBV::CoilSpring>(obj);
  iconFile="coilspring.svg";
  setIcon(0, Utils::QIconCached(iconFile));

  //h5 dataset
  int rows=coilSpring->getRows();
  double dt;
  vector<OpenMBV::Float> data0;
  if(rows>=1) data0=coilSpring->getRow(0);
  if(rows>=2) dt=coilSpring->getRow(1)[0]-data0[0]; else dt=0;
  resetAnimRange(rows, dt);

  double R=coilSpring->getSpringRadius();
  double r=coilSpring->getCrossSectionRadius();
  N=coilSpring->getNumberOfCoils();
  if(r<0) {
    if(coilSpring->getType()==OpenMBV::CoilSpring::polyline)
      r=2;
    else
      r=R/7;
  }

  // read XML
  scaleValue=coilSpring->getScaleFactor();
  nominalLength=coilSpring->getNominalLength();
  if(nominalLength<0) {
    if(data0.size()>0) {
      SbVec3f distance(data0[4]-data0[1],data0[5]-data0[2],data0[6]-data0[3]);
      nominalLength=distance.length();
    }
    else
      nominalLength=r*N*4;
  }

  // create so
  // body
  fromPoint=new SoTranslation;
  soSep->addChild(fromPoint);
  rotation=new SoRotation;
  soSep->addChild(rotation);  

  switch(coilSpring->getType()) {
    case OpenMBV::CoilSpring::tube: {
      tube = make_unique<ExtrusionCardan>();
      auto contour = make_shared<std::vector<std::shared_ptr<OpenMBV::PolygonPoint>>>();
      for(int i=0;i<iCircSegments;i++) // clockwise in local coordinate system
        contour->emplace_back(OpenMBV::PolygonPoint::create(r*cos(i*2.*M_PI/iCircSegments), -r*sin(i*2.*M_PI/iCircSegments), 0));
      tube->init(int(numberOfSpinePointsPerCoil*N)+1, contour,
                 1.0, false,
                 soSep, soOutLineSep);
      // initialise spine 
      spine.resize(6*(int(numberOfSpinePointsPerCoil*N)+1)+1);
      for(int i=0;i<=numberOfSpinePointsPerCoil*N;i++) {
        float alpha = i*N*2.*M_PI/numberOfSpinePointsPerCoil/N;
        spine[6*i+1] = R*cos(alpha);
        spine[6*i+2] = R*sin(alpha);
        spine[6*i+3] = 0;
        spine[6*i+4] = 0;
        spine[6*i+5] = 0;
        spine[6*i+6] = alpha;
      }
      tube->setCardanWrtWorldSpine(spine);
      break;
    }
    case OpenMBV::CoilSpring::tubeShader: {
      MainWindow::getInstance()->addPickUpdate(this);
      tubeShader = make_unique<CoilSpringShader>();
      tubeShader->init(R, N, numberOfSpinePointsPerCoil, int(numberOfSpinePointsPerCoil*N)+1, iCircSegments, r, mat, soSep);
      tubeShader->updateData(nominalLength);
      tubeShader->pickUpdate();
      tubeShader->pickUpdateRestore();
      break;
    }
    case OpenMBV::CoilSpring::scaledTube: {
      auto *scaledTubeSep=new SoSeparator;
      soSep->addChild(scaledTubeSep);
      scale=new SoScale;
      scaledTubeSep->addChild(scale);
      auto *scaledExtrusion=new SoVRMLExtrusion;
      scaledTubeSep->addChild(scaledExtrusion);
      // cross section
      scaledExtrusion->crossSection.setNum(iCircSegments+1);
      SbVec2f *scs = scaledExtrusion->crossSection.startEditing();
      for(int i=0;i<iCircSegments;i++) // clockwise in local coordinate system
        scs[i]=SbVec2f(r*cos(i*2.*M_PI/iCircSegments), -r*sin(i*2.*M_PI/iCircSegments));
      scs[iCircSegments]=scs[0]; // close cross section: uses exact the same point: helpfull for "binary space partitioning container"
      scaledExtrusion->crossSection.finishEditing();
      scaledExtrusion->crossSection.setDefault(FALSE);
      // initialise spine 
      scaledSpine = new float[3*(int(numberOfSpinePointsPerCoil*N)+1)];
      for(int i=0;i<=numberOfSpinePointsPerCoil*N;i++) {
        scaledSpine[3*i]= R*cos(i*N*2.*M_PI/numberOfSpinePointsPerCoil/N);
        scaledSpine[3*i+1]= R*sin(i*N*2.*M_PI/numberOfSpinePointsPerCoil/N);
        scaledSpine[3*i+2] = i*nominalLength/numberOfSpinePointsPerCoil/N;
      }
      scaledExtrusion->spine.setValuesPointer(int(numberOfSpinePointsPerCoil*N+1),scaledSpine);
      scaledExtrusion->spine.setDefault(FALSE);
      // additional flags
      scaledExtrusion->solid=TRUE; // backface culling
      scaledExtrusion->convex=TRUE; // only convex polygons included in visualisation
      scaledExtrusion->ccw=TRUE; // vertex ordering counterclockwise?
      scaledExtrusion->beginCap=TRUE; // front side at begin of the spine
      scaledExtrusion->endCap=TRUE; // front side at end of the spine
      scaledExtrusion->creaseAngle=1.5; // angle below which surface normals are drawn smooth (always smooth, except begin/end cap => < 90deg)
      break;
    }
    case OpenMBV::CoilSpring::polyline: {
      auto *polylineSep=new SoSeparator;
      soSep->addChild(polylineSep);
      scale=new SoScale;
      polylineSep->addChild(scale);
      auto *ds=new SoDrawStyle;
      polylineSep->addChild(ds);
      ds->lineWidth.setValue(r);
      auto *polylineCoord=new SoCoordinate3;
      polylineSep->addChild(polylineCoord);
      auto *polyline=new SoLineSet;
      polylineSep->addChild(polyline);
      polyline->numVertices.setValue(int(numberOfSpinePointsPerCoil*N)+1);
      // initialise spine 
      scaledSpine = new float[3*(int(numberOfSpinePointsPerCoil*N)+1)];
      for(int i=0;i<=numberOfSpinePointsPerCoil*N;i++) {
        scaledSpine[3*i]= R*cos(i*N*2.*M_PI/numberOfSpinePointsPerCoil/N);
        scaledSpine[3*i+1]= R*sin(i*N*2.*M_PI/numberOfSpinePointsPerCoil/N);
        scaledSpine[3*i+2] = i*nominalLength/numberOfSpinePointsPerCoil/N;
      }
      polylineCoord->point.setValuesPointer(int(numberOfSpinePointsPerCoil*N+1),scaledSpine);
      break;
    }
  }
}

void CoilSpring::createProperties() {
  DynamicColoredBody::createProperties();

  // GUI editors
  if(!clone) {
    properties->updateHeader();
    auto *typeEditor=new ComboBoxEditor(properties, QIcon(), "Type", {
      make_tuple(OpenMBV::CoilSpring::tube,       "Tube",        QIcon(), "CoilSpring::type::tube"),
      make_tuple(OpenMBV::CoilSpring::scaledTube, "Scaled tube", QIcon(), "CoilSpring::type::scaledTube"),
      make_tuple(OpenMBV::CoilSpring::polyline,   "Polyline",    QIcon(), "CoilSpring::type::polyline"),
      make_tuple(OpenMBV::CoilSpring::tubeShader, "Tube shader", QIcon(), "CoilSpring::type::tubeShader"),
    });
    typeEditor->setOpenMBVParameter(coilSpring, &OpenMBV::CoilSpring::getType, &OpenMBV::CoilSpring::setType);
    properties->addPropertyActionGroup(typeEditor->getActionGroup());

    auto *numberOfCoilsEditor=new FloatEditor(properties, QIcon(), "Number of coils");
    numberOfCoilsEditor->setRange(0, DBL_MAX);
    numberOfCoilsEditor->setOpenMBVParameter(coilSpring, &OpenMBV::CoilSpring::getNumberOfCoils, &OpenMBV::CoilSpring::setNumberOfCoils);

    auto *springRadiusEditor=new FloatEditor(properties, QIcon(), "Coil spring radius");
    springRadiusEditor->setRange(0, DBL_MAX);
    springRadiusEditor->setOpenMBVParameter(coilSpring, &OpenMBV::CoilSpring::getSpringRadius, &OpenMBV::CoilSpring::setSpringRadius);

    auto *crossSectionRadiusEditor=new FloatEditor(properties, QIcon(), "Cross section radius");
    crossSectionRadiusEditor->setRange(0, DBL_MAX);
    crossSectionRadiusEditor->setOpenMBVParameter(coilSpring, &OpenMBV::CoilSpring::getCrossSectionRadius, &OpenMBV::CoilSpring::setCrossSectionRadius);

    auto *nominalLengthEditor=new FloatEditor(properties, QIcon(), "Nominal length");
    nominalLengthEditor->setRange(0, DBL_MAX);
    nominalLengthEditor->setOpenMBVParameter(coilSpring, &OpenMBV::CoilSpring::getNominalLength, &OpenMBV::CoilSpring::setNominalLength);

    auto *scaleFactorEditor=new FloatEditor(properties, QIcon(), "Scale factor");
    scaleFactorEditor->setOpenMBVParameter(coilSpring, &OpenMBV::CoilSpring::getScaleFactor, &OpenMBV::CoilSpring::setScaleFactor);
  }
}

CoilSpring::~CoilSpring() {
  MainWindow::getInstance()->removePickUpdate(this);
  delete[]scaledSpine;
}

QString CoilSpring::getInfo() {
  float x, y, z;
  fromPoint->translation.getValue().getValue(x,y,z);
  float sx, sy, sz=0;
  if(coilSpring->getType()!=OpenMBV::CoilSpring::tube && coilSpring->getType()!=OpenMBV::CoilSpring::tubeShader)
    scale->scaleFactor.getValue().getValue(sx, sy, sz);
  return DynamicColoredBody::getInfo()+
         QString("<hr width=\"10000\"/>")+
         QString("<b>From point:</b> %1, %2, %3<br/>").arg(x).arg(y).arg(z)+
         QString("<b>Length:</b> %1").arg(coilSpring->getType()==OpenMBV::CoilSpring::tube ||
                                          coilSpring->getType()==OpenMBV::CoilSpring::tubeShader?
                                          len:
                                          sz*nominalLength);
}

double CoilSpring::update() {
  // read from hdf5
  int frame=MainWindow::getInstance()->getFrame()[0];
  auto data=coilSpring->getRow(frame);

  // translation / rotation
  fromPoint->translation.setValue(data[1],data[2],data[3]);
  SbVec3f distance(data[4]-data[1],data[5]-data[2],data[6]-data[3]);
  len = distance.length() * scaleValue;
  rotation->rotation.setValue(SbRotation(SbVec3f(0,0,1),distance));

  switch(coilSpring->getType()) {
    case OpenMBV::CoilSpring::tube:
      // tube 
      for(int i=0;i<=int(numberOfSpinePointsPerCoil*N);i++) {
        spine[6*i+3] = i*len/numberOfSpinePointsPerCoil/N;
        float alpha = i*N*2.*M_PI/numberOfSpinePointsPerCoil/N;
        double R=coilSpring->getSpringRadius();
        float n = atan( len / ( N*2.*M_PI*R ) );
        // spine[6*i+4:6] = Utils::rotation2Cardan(Utils::cardan2Rotation(SbVec3f(0,0,alpha))*Utils::cardan2Rotation(SbVec3f(n,0,0)));
        float sinn = sin(n);
        float cosn = cos(n);
        float sinalpha = sin(alpha);
        float cosalpha = cos(alpha);
        spine[6*i+4] = atan2(sinn*cosalpha, cosn);
        spine[6*i+5] = asin(sinalpha*sinn);
        spine[6*i+6] = atan2(sinalpha*cosn, cosalpha);
      }
      tube->setCardanWrtWorldSpine(spine, coilSpring->getUpdateNormals());
      break;
    case OpenMBV::CoilSpring::tubeShader:
      // tube shader
      tubeShader->updateData(len);
      break;
    case OpenMBV::CoilSpring::scaledTube:
    case OpenMBV::CoilSpring::polyline:
      scale->scaleFactor.setValue(1,1,len/nominalLength);
      break;
  }
  
  // color
  if(diffuseColor[0]<0) setColor(data[7]);

  return data[0];
}

void CoilSpring::pickUpdate() {
  tubeShader->pickUpdate();
}

void CoilSpring::pickUpdateRestore() {
  tubeShader->pickUpdateRestore();
}

void CoilSpringShader::pickUpdate() {
  sepNoPickNoBBox->skipPick.setValue(false);

  // update coords
  auto c = vertex->point.startEditing();
    for(int spIdx=0; spIdx<Nsp; ++spIdx) {
      float alpha = spIdx*2.*M_PI/numberOfSpinePointsPerCoil;
      SbVec3f r_(
        R*cos(alpha),
        R*sin(alpha),
        spIdx*length->value.getValue()/numberOfSpinePointsPerCoil/N
      );
      float n = atan( length->value.getValue() / ( N*2.*M_PI*R ) );
      // angle = rotation2Cardan(cardan2Rotation(0,0,alpha)*cardan2Rotation(n,0,0));
      float sinn = sin(n);
      float cosn = cos(n);
      float sinalpha = sin(alpha);
      float cosalpha = cos(alpha);
      SbVec3f angle(
        atan2(sinn*cosalpha, cosn),
        asin(sinalpha*sinn),
         atan2(sinalpha*cosn, cosalpha)
      );
      SbRotation T(Utils::cardan2Rotation(angle));
      for(int csIdx=0; csIdx<Ncs; ++csIdx) {
        float arg = csIdx*2.*M_PI/Ncs;
        SbVec3f nsp(r*cos(arg), 0, -r*sin(arg));
        SbVec3f T_nsp;
        T.multVec(nsp, T_nsp);
        c[csIdx+Ncs*spIdx] = r_ + T_nsp;
      }
    }
  vertex->point.finishEditing();
}

void CoilSpringShader::pickUpdateRestore() {
  sepNoPickNoBBox->skipPick.setValue(true);
}

namespace {
  string S(int x) {
    return to_string(x);
  };
  string S(double x) {
    return boost::lexical_cast<string>(static_cast<float>(x));
  };
}

void CoilSpringShader::init(double R_, double N_, int numberOfSpinePointsPerCoil_, int Nsp_, int Ncs_, double r_, SoMaterial *mat, SoSeparator *soSep) {
  R = R_;
  r = r_;
  N = N_;
  numberOfSpinePointsPerCoil = numberOfSpinePointsPerCoil_;
  Nsp = Nsp_;
  Ncs = Ncs_;

  vector<SbVec2f> circle(Ncs);
  for(int i=0;i<Ncs;i++) // clockwise in local coordinate system
    circle[i] = SbVec2f(r*cos(i*2.*M_PI/Ncs), -r*sin(i*2.*M_PI/Ncs));

  length = new SoShaderParameter1f;
  soSep->addChild(length);
  length->setName("openmbv_coilspring_length");

  static const string ivFilename((boost::dll::program_location().parent_path().parent_path()/"share"/"openmbv"/"coilspring.iv").string());
  ifstream ivFile(ivFilename);
  std::stringstream buf;
  buf << ivFile.rdbuf();
  string ivContent(buf.str());

  string NcsStr;
  for(int i=0; i<Ncs; ++i)
    NcsStr+=" "+S(i);

  string nspStr2;
  for(int i=0; i<Ncs; ++i) {
    if(i%5==0) nspStr2+="\n";
    nspStr2+=" "+S(circle[i][0])+" 0 "+S(circle[i][1]);
  }

  map<string, string> replace {
    { "R"                         , S(R) },
    { "r"                         , S(r) },
    { "N"                         , S(N) },
    { "numberOfSpinePointsPerCoil", to_string(numberOfSpinePointsPerCoil) },
    { "Nsp"                       , S(Nsp) },
    { "Ncs"                       , S(Ncs) },
    { "nspStr2"                   ,   nspStr2 },
    { "NcsStr"                    ,   NcsStr },
    { "startIndex1"               , S(6*(Nsp-1)+1) },
    { "startIndex2"               , S(6*(Nsp-1)+4) },
    { "2Rpr"                      , S(2*(R+r)) },
  };
  ivContent = replaceKeys(ivContent, replace);

  static bool OPENMBV_DUMP_COILSPRING_IV=getenv("OPENMBV_DUMP_COILSPRING_IV")!=nullptr;
  if(OPENMBV_DUMP_COILSPRING_IV) {
    static int i=0;
    ofstream f("coilspring_"+S(++i)+".iv");
    f<<ivContent;
  }

  auto soIv = Utils::SoDBreadAllContentCached(ivContent, {/*no cache*/}, [this](SoInput& in) {
    in.addReference("openmbv_coilspring_length", length);
  });
  if(!soIv)
    return;
  soSep->addChild(soIv);

  coords=static_cast<SoCoordinate3*>(Utils::getChildNodeByName(soIv, "openmbv_coilspring_coords"));
  endCap1Trans=static_cast<SoTransform*>(Utils::getChildNodeByName(soIv, "endCap1Trans"));
  endCap2Trans=static_cast<SoTransform*>(Utils::getChildNodeByName(soIv, "endCap2Trans"));

  vertex = static_cast<SoCoordinate3*>(Utils::getChildNodeByName(soSep, "openmbv_coilspring_coords"));
  sepNoPickNoBBox = static_cast<SepNoPickNoBBox*>(Utils::getChildNodeByName(soSep, "openmbv_coilspring_sepnopicknobbox"));

  // set coords: only the numbers, values are set later
  coords->point.setNum(Nsp*Ncs);
  // set normals: only the numbers, values are set by the shader
  auto normal=static_cast<SoNormal*>(Utils::getChildNodeByName(soIv, "openmbv_coilspring_normals"));
  normal->vector.setNum(Nsp*Ncs);
  // set vertex attributes: constant values
  auto vertAttr = static_cast<SoMFFloat*>(static_cast<SoVertexAttribute*>(
                    Utils::getChildNodeByName(soIv, "openmbv_coilspring_vertattr"))->getValuesField());
  vertAttr->setNum(Nsp*Ncs);
  float* va=vertAttr->startEditing();
    for(int i=0; i<Nsp*Ncs; ++i) {
      if(static_cast<int>(static_cast<float>(i))!=i) {
        auto msg("Due to restrictions in Coin we need to convert the vertex ID 'int' to a 'float' on the CPU\n"
                 "and than back to 'int' on the GPU. The number of vertices in this CoilSpring are too large\n"
                 "for this conversion. (ID="+to_string(i)+")\n"
                 "Please use less numberOfCoils or switch to 'tube' or set the envvar\n"
                 "'OPENMBV_DISABLE_SHADER' which will switch to 'tube' automatically.\n"
                 "Exiting now");
        QMessageBox::critical(nullptr, "Critical Error", msg.c_str());
        throw runtime_error(msg);
      }
      va[i] = i;
    }
  vertAttr->finishEditing();
  // set coord indices: constant values
  auto coordIndex = static_cast<SoIndexedFaceSet*>(Utils::getChildNodeByName(soIv, "openmbv_coilspring_coordindex"));
  coordIndex->coordIndex.setNum((Nsp-1)*Ncs*5);
  auto ci = coordIndex->coordIndex.startEditing();
  int i=0;
  for(int spIdx=0; spIdx<Nsp-1; ++spIdx) {
    for(int csIdx=0; csIdx<Ncs; ++csIdx) {
      int nIdx = spIdx*Ncs+csIdx;
      ci[i++] = nIdx;
      ci[i++] = nIdx+(csIdx<Ncs-1 ? 1 :1-Ncs);
      ci[i++] = nIdx+(csIdx<Ncs-1 ? 1 :1-Ncs)+Ncs;
      ci[i++] = nIdx+Ncs;
      ci[i++] = -1;
    }
  }
  coordIndex->coordIndex.finishEditing();
}

void CoilSpringShader::updateData(double len) {
  length->value.setValue(len);

  auto calcrT = [this, len](int i) {
    float alpha = i*N*2.*M_PI/numberOfSpinePointsPerCoil/N;
    SbVec3f r(
      R*cos(alpha),
      R*sin(alpha),
      i*len/numberOfSpinePointsPerCoil/N
    );
    float n = atan( len / ( N*2.*M_PI*R ) );
    // spine[6*i+4:6] = Utils::rotation2Cardan(Utils::cardan2Rotation(SbVec3f(0,0,alpha))*Utils::cardan2Rotation(SbVec3f(n,0,0)));
    float sinn = sin(n);
    float cosn = cos(n);
    float sinalpha = sin(alpha);
    float cosalpha = cos(alpha);
    SbVec3f angle(
      atan2(sinn*cosalpha, cosn),
      asin(sinalpha*sinn),
      atan2(sinalpha*cosn, cosalpha)
    );
    return make_pair(r, Utils::cardan2Rotation(angle).invert());
  };
  auto [r1, T1] = calcrT(0);
  endCap1Trans->translation.setValue(r1);
  endCap1Trans->rotation.setValue(T1);
  auto [r2, T2] = calcrT(numberOfSpinePointsPerCoil*N);
  endCap2Trans->translation.setValue(r2);
  endCap2Trans->rotation.setValue(T2);
}

}
