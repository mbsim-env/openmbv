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

#include <config.h>
#include "utils.h"
#include <Inventor/nodes/SoLineSet.h>
#include <Inventor/nodes/SoComplexity.h>
#include <Inventor/nodes/SoPerspectiveCamera.h>
#include <Inventor/nodes/SoOrthographicCamera.h>
#include "SoSpecial.h"
#include "mainwindow.h"
#include "mytouchwidget.h"
#include <iostream>
#include <QTapAndHoldGesture>
#include <QScroller>
#include <QDialog>
#include <QPushButton>
#include <QLineEdit>
#include <QMessageBox>
#include <QGridLayout>
#include <QComboBox>
#include <QLabel>
#include <QBitArray>
#include <boost/dll.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;

namespace OpenMBVGUI {

unordered_map<string, Utils::SoDeleteSeparator> Utils::ivBodyCache;
unordered_map<string, QIcon> Utils::iconCache;
bool Utils::initialized=false;

void Utils::initialize() {
  if(initialized) return;
  initialized=true;

  // tess
  gluTessCallback(tess, GLU_TESS_BEGIN_DATA, (void (CALLMETHOD *)())tessBeginCB);
  gluTessCallback(tess, GLU_TESS_VERTEX, (void (CALLMETHOD *)())tessVertexCB);
  gluTessCallback(tess, GLU_TESS_END, (void (CALLMETHOD *)())tessEndCB);
}

void Utils::deinitialize() {
  if(!initialized) return;
  initialized=false;

  iconCache.clear();
}

const QIcon& Utils::QIconCached(string filename) {
  // fix relative filename
  if(filename[0]!=':' && filename[0]!='/')
    filename=getIconPath()+"/"+filename;
  
  pair<unordered_map<string, QIcon>::iterator, bool> ins=iconCache.insert(pair<string, QIcon>(filename, QIcon()));
  if(ins.second)
    return ins.first->second=QIcon(filename.c_str());
  return ins.first->second;
}

SoSeparator* Utils::SoDBreadAllCached(const string &filename) {
  auto ins=ivBodyCache.emplace(filename, SoDeleteSeparator());
  if(ins.second) {
    SoInput in;
    if(in.openFile(filename.c_str(), true)) { // if file can be opened, read it
      ins.first->second.s=SoDB::readAll(&in); // stored in a global cache => false positive in valgrind
      ins.first->second.s->ref(); // increment reference count to prevent a delete for cached entries
      return ins.first->second.s;
    }
    else { // open failed, remove from cache and return a empty IV
      QString str("Unable to find IV file %1."); str=str.arg(filename.c_str());
      MainWindow::getInstance()->statusBar()->showMessage(str);
      msgStatic(Warn)<<str.toStdString()<<endl;
      ivBodyCache.erase(ins.first);
      return new SoSeparator;
    }
  }
  return ins.first->second.s;
}

// convenience: create frame so
SoSeparator* Utils::soFrame(double size, double offset, bool pickBBoxAble, SoScale *&scale,
                            const SbColor &xCol, const SbColor &yCol, const SbColor &zCol) {
  SoSeparator *sep;
  if(pickBBoxAble)
    sep=new SoSeparator;
  else
    sep=new SoSepNoPickNoBBox;

  SoBaseColor *col;
  SoLineSet *line;

  // coordinates
  scale=new SoScale;
  sep->addChild(scale);
  scale->scaleFactor.setValue(size, size, size);
  auto *coord=new SoCoordinate3;
  sep->addChild(coord);
  coord->point.set1Value(0, -1.0/2+offset*1.0/2, 0, 0);
  coord->point.set1Value(1, +1.0/2+offset*1.0/2, 0, 0);
  coord->point.set1Value(2, 0, -1.0/2+offset*1.0/2, 0);
  coord->point.set1Value(3, 0, +1.0/2+offset*1.0/2, 0);
  coord->point.set1Value(4, 0, 0, -1.0/2+offset*1.0/2);
  coord->point.set1Value(5, 0, 0, +1.0/2+offset*1.0/2);

  // x-axis
  col=new SoBaseColor;
  col->rgb=xCol;
  sep->addChild(col);
  line=new SoLineSet;
  line->startIndex.setValue(0);
  line->numVertices.setValue(2);
  sep->addChild(line);

  // y-axis
  col=new SoBaseColor;
  col->rgb=yCol;
  sep->addChild(col);
  line=new SoLineSet;
  line->startIndex.setValue(2);
  line->numVertices.setValue(2);
  sep->addChild(line);

  // z-axis
  col=new SoBaseColor;
  col->rgb=zCol;
  sep->addChild(col);
  line=new SoLineSet;
  line->startIndex.setValue(4);
  line->numVertices.setValue(2);
  sep->addChild(line);

  return sep;
}

SbRotation Utils::cardan2Rotation(const SbVec3f &c) {
  float a, b, g;
  c.getValue(a,b,g);
  return SbMatrix(
    cos(b)*cos(g),
    -cos(b)*sin(g),
    sin(b),
    0.0,

    cos(a)*sin(g)+sin(a)*sin(b)*cos(g),
    cos(a)*cos(g)-sin(a)*sin(b)*sin(g),
    -sin(a)*cos(b),
    0.0,

    sin(a)*sin(g)-cos(a)*sin(b)*cos(g),
    sin(a)*cos(g)+cos(a)*sin(b)*sin(g),
    cos(a)*cos(b),
    0.0,

    0.0,
    0.0,
    0.0,
    1.0
  );
}

SbVec3f Utils::rotation2Cardan(const SbRotation& R) {
  SbMatrix M;
  R.getValue(M);
  float a, b, g;
  b=asin(M[0][2]);
  double nenner=cos(b);
  if (nenner>1e-10) {
    a=atan2(-M[1][2],M[2][2]);
    g=atan2(-M[0][1],M[0][0]);
  } else {
    a=0;
    g=atan2(M[1][0],M[1][1]);
  }
  return {a,b,g};
}

// for tess
GLUtesselator *Utils::tess=gluNewTess();
GLenum Utils::tessType;
int Utils::tessNumVertices;
SoTriangleStripSet *Utils::tessTriangleStrip;
SoIndexedFaceSet *Utils::tessTriangleFan;
SoCoordinate3 *Utils::tessCoord;

// tess
void Utils::tessBeginCB(GLenum type, void *data) {
  auto *parent=(SoGroup*)data;
  tessType=type;
  tessNumVertices=0;
  tessCoord=new SoCoordinate3;
  parent->addChild(tessCoord);
  if(tessType==GL_TRIANGLES || tessType==GL_TRIANGLE_STRIP) {
    tessTriangleStrip=new SoTriangleStripSet;
    parent->addChild(tessTriangleStrip);
  }
  if(tessType==GL_TRIANGLE_FAN) {
    tessTriangleFan=new SoIndexedFaceSet;
    parent->addChild(tessTriangleFan);
  }
}

void Utils::tessVertexCB(GLdouble *vertex) {
  tessCoord->point.set1Value(tessNumVertices++, vertex[0], vertex[1], vertex[2]);
}

void Utils::tessEndCB() {
  if(tessType==GL_TRIANGLES)
    for(int i=0; i<tessNumVertices/3; i++)
      tessTriangleStrip->numVertices.set1Value(i, 3);
  if(tessType==GL_TRIANGLE_STRIP)
    tessTriangleStrip->numVertices.set1Value(0, tessNumVertices);
  if(tessType==GL_TRIANGLE_FAN) {
    int j=0;
    for(int i=0; i<tessNumVertices-2; i++) {
      tessTriangleFan->coordIndex.set1Value(j++, 0);
      tessTriangleFan->coordIndex.set1Value(j++, i+1);
      tessTriangleFan->coordIndex.set1Value(j++, i+2);
      tessTriangleFan->coordIndex.set1Value(j++, -1);
    }
  }
}

std::shared_ptr<OpenMBV::Object> Utils::createObjectEditor(const vector<FactoryElement> &factory, const vector<string> &existingNames, const string &title) {
  bool exist;
  int i=0;
  string name;
  do {
    i++;
    stringstream str;
    str<<"Untitled"<<i;
    name=str.str();
    exist=false;
    for(const auto & existingName : existingNames)
      if(existingName==name) {
        exist=true;
        break;
      }
  } while(exist);

  QDialog dialog;
  dialog.setWindowTitle(title.c_str());
  auto *layout=new QGridLayout();
  dialog.setLayout(layout);

  layout->addWidget(new QLabel("Type:"), 0, 0);
  auto *cb=new QComboBox();
  layout->addWidget(cb, 0, 1);
  for(const auto & i : factory)
    cb->addItem(get<0>(i), get<1>(i).c_str());

  layout->addWidget(new QLabel("Name:"), 1, 0);
  auto *lineEdit=new QLineEdit();
  layout->addWidget(lineEdit, 1, 1);
  lineEdit->setText(name.c_str());

  auto *cancel=new QPushButton("Cancel");
  layout->addWidget(cancel, 2, 0);
  QObject::connect(cancel, &QPushButton::released, &dialog, &QDialog::reject);
  auto *ok=new QPushButton("OK");
  layout->addWidget(ok, 2, 1);
  ok->setDefault(true);
  QObject::connect(ok, &QPushButton::released, &dialog, &QDialog::accept);

  bool unique;
  do {
    if(dialog.exec()!=QDialog::Accepted) return {};
    unique=true;
    for(const auto & existingName : existingNames)
      if(existingName==lineEdit->text().toStdString()) {
        QMessageBox::information(nullptr, "Information", "The entered name already exists!");
        unique=false;
        break;
      }
  } while(!unique);

  std::shared_ptr<OpenMBV::Object> obj=get<2>(factory[cb->currentIndex()])();
  obj->setName(lineEdit->text().toStdString());
  return obj;
}

namespace {
  boost::filesystem::path sharePath(boost::dll::program_location().parent_path().parent_path()/"share");
}

string Utils::getIconPath() {
  return (sharePath/"openmbv"/"icons").string();
}

string Utils::getXMLDocPath() {
  return (sharePath/"mbxmlutils"/"doc").string();
}

string Utils::getDocPath() {
  return (sharePath/"openmbv"/"doc").string();
}

void Utils::enableTouch(QWidget *w) {
  QScroller::grabGesture(w, QScroller::TouchGesture);
  QScrollerProperties scrollerProps;
  scrollerProps.setScrollMetric(QScrollerProperties::VerticalOvershootPolicy, QScrollerProperties::OvershootAlwaysOff);
  scrollerProps.setScrollMetric(QScrollerProperties::HorizontalOvershootPolicy, QScrollerProperties::OvershootAlwaysOff);
  QScroller::scroller(w)->setScrollerProperties(scrollerProps);
}

IgnoreWheelEventFilter IgnoreWheelEventFilter::instance;

bool IgnoreWheelEventFilter::eventFilter(QObject *watched, QEvent *event) {
  if(event->type()==QEvent::Wheel) { return true; }
  return QObject::eventFilter(watched, event);
}

AppSettings::AppSettings() : qSettings(format, scope, organization, application), setting(AS::SIZE) {
  setting[hdf5RefreshDelta]={"mainwindow/hdf5/hdf5RefreshDelta", 500};
  setting[cameraType]={"mainwindow/sceneGraph/cameraType", 0};
  setting[stereoType]={"mainwindow/sceneGraph/stereoType", 0};
  setting[stereoOffset]={"mainwindow/sceneGraph/stereoOffset", 0.1};
  setting[tapAndHoldTimeout]={"mainwindow/manipulate3d/tapAndHoldTimeout", 700};
  setting[outlineShilouetteEdgeLineWidth]={"mainwindow/sceneGraph/outlineShilouetteEdgeLineWidth", 1.0};
  setting[outlineShilouetteEdgeLineColor]={"mainwindow/sceneGraph/outlineShilouetteEdgeLineColor", QColor(0,0,0)};
  setting[boundingBoxLineWidth]={"mainwindow/sceneGraph/boundingBoxLineWidth", 2.0};
  setting[boundingBoxLineColor]={"mainwindow/sceneGraph/boundingBoxLineColor", QColor(0,190,0)};
  setting[highlightLineWidth]={"mainwindow/sceneGraph/highlightLineWidth", 3.0};
  setting[highlightLineColor]={"mainwindow/sceneGraph/highlightLineColor", QColor(0,255,255)};
  setting[complexityType]={"mainwindow/sceneGraph/complexityType", 1};
  setting[complexityValue]={"mainwindow/sceneGraph/complexityValue", 0.2};
  setting[topBackgroudColor]={"mainwindow/sceneGraph/topBackgroudColor", QColor(210,210,254)};
  setting[bottomBackgroundColor]={"mainwindow/sceneGraph/bottomBackgroundColor", QColor(89,89,152)};
  setting[anglePerKeyPress]={"mainwindow/manipulate3d/anglePerKeyPress", 5};
  setting[speedChangeFactor]={"mainwindow/manipulate3d/speedChangeFactor", 1.1};
  setting[shortAniTime]={"mainwindow/manipulate3d/shortAniTime", 300};
  setting[mainwindow_geometry]={"mainwindow/geometry", QVariant()};
  setting[mainwindow_state]={"mainwindow/state", QVariant()};
  setting[settingsDialog_geometry]={"settingsDialog/geometry", QVariant()};
  setting[exportdialog_resolutionfactor]={"exportdialog/resolutionfactor", 1.0};
  setting[exportdialog_usescenecolor]={"exportdialog/usescenecolor", true};
  setting[exportdialog_fps]={"exportdialog/fps", 25.0};
  setting[exportdialog_filename_png]={"exportdialog/filename/png", "openmbv.png"};
  setting[exportdialog_filename_video]={"exportdialog/filename/video", "openmbv.webm"};
  setting[exportdialog_bitrate]={"exportdialog/bitrate", 2048};
  setting[exportdialog_videocmd]={"exportdialog/videocmd",
    "ffmpeg -framerate %F -i openmbv_%06d.png -c:v libvpx-vp9 -b:v %B -pass 1 -f null /dev/null && "
    "ffmpeg -framerate %F -i openmbv_%06d.png -c:v libvpx-vp9 -b:v %B -pass 2 %O"};
  setting[propertydialog_geometry]={"propertydialog/geometry", QVariant()};
  setting[dialogstereo_geometry]={"dialogstereo/geometry", QVariant()};
  setting[mouseCursorSize]={"mainwindow/manipulate3d/mouseCursorSize", 5.0};
  using MA=MyTouchWidget::MoveAction;
  setting[mouseNoneLeftMoveAction]=        {"mainwindow/manipulate3d/mouseNoneLeftMoveAction", static_cast<int>(MA::RotateAboutSySx)};
  setting[mouseShiftLeftMoveAction]=       {"mainwindow/manipulate3d/mouseShiftLeftMoveAction", static_cast<int>(MA::None)};
  setting[mouseCtrlLeftMoveAction]=        {"mainwindow/manipulate3d/mouseCtrlLeftMoveAction", static_cast<int>(MA::RotateAboutSz)};
  setting[mouseAltLeftMoveAction]=         {"mainwindow/manipulate3d/mouseAltLeftMoveAction", static_cast<int>(MA::None)};
  setting[mouseShiftCtrlLeftMoveAction]=   {"mainwindow/manipulate3d/mouseShiftCtrlLeftMoveAction", static_cast<int>(MA::None)};
  setting[mouseShiftAltLeftMoveAction]=    {"mainwindow/manipulate3d/mouseShiftAltLeftMoveAction", static_cast<int>(MA::None)};
  setting[mouseCtrlAltLeftMoveAction]=     {"mainwindow/manipulate3d/mouseCtrlAltLeftMoveAction", static_cast<int>(MA::None)};
  setting[mouseShiftCtrlAltLeftMoveAction]={"mainwindow/manipulate3d/mouseShiftCtrlAltLeftMoveAction", static_cast<int>(MA::None)};
  setting[mouseNoneRightMoveAction]=        {"mainwindow/manipulate3d/mouseNoneRightMoveAction", static_cast<int>(MA::Translate)};
  setting[mouseShiftRightMoveAction]=       {"mainwindow/manipulate3d/mouseShiftRightMoveAction", static_cast<int>(MA::None)};
  setting[mouseCtrlRightMoveAction]=        {"mainwindow/manipulate3d/mouseCtrlRightMoveAction", static_cast<int>(MA::None)};
  setting[mouseAltRightMoveAction]=         {"mainwindow/manipulate3d/mouseAltRightMoveAction", static_cast<int>(MA::None)};
  setting[mouseShiftCtrlRightMoveAction]=   {"mainwindow/manipulate3d/mouseShiftCtrlRightMoveAction", static_cast<int>(MA::None)};
  setting[mouseShiftAltRightMoveAction]=    {"mainwindow/manipulate3d/mouseShiftAltRightMoveAction", static_cast<int>(MA::None)};
  setting[mouseCtrlAltRightMoveAction]=     {"mainwindow/manipulate3d/mouseCtrlAltRightMoveAction", static_cast<int>(MA::None)};
  setting[mouseShiftCtrlAltRightMoveAction]={"mainwindow/manipulate3d/mouseShiftCtrlAltRightMoveAction", static_cast<int>(MA::None)};
  setting[mouseNoneMidMoveAction]=        {"mainwindow/manipulate3d/mouseNoneMidMoveAction", static_cast<int>(MA::Zoom)};
  setting[mouseShiftMidMoveAction]=       {"mainwindow/manipulate3d/mouseShiftMidMoveAction", static_cast<int>(MA::None)};
  setting[mouseCtrlMidMoveAction]=        {"mainwindow/manipulate3d/mouseCtrlMidMoveAction", static_cast<int>(MA::CameraFocalDistance)};
  setting[mouseAltMidMoveAction]=         {"mainwindow/manipulate3d/mouseAltMidMoveAction", static_cast<int>(MA::None)};
  setting[mouseShiftCtrlMidMoveAction]=   {"mainwindow/manipulate3d/mouseShiftCtrlMidMoveAction", static_cast<int>(MA::None)};
  setting[mouseShiftAltMidMoveAction]=    {"mainwindow/manipulate3d/mouseShiftAltMidMoveAction", static_cast<int>(MA::None)};
  setting[mouseCtrlAltMidMoveAction]=     {"mainwindow/manipulate3d/mouseCtrlAltMidMoveAction", static_cast<int>(MA::None)};
  setting[mouseShiftCtrlAltMidMoveAction]={"mainwindow/manipulate3d/mouseShiftCtrlAltMidMoveAction", static_cast<int>(MA::None)};
  setting[mouseNoneWheelAction]=        {"mainwindow/manipulate3d/mouseNoneWheelAction", static_cast<int>(MA::ChangeFrame)};
  setting[mouseShiftWheelAction]=       {"mainwindow/manipulate3d/mouseShiftWheelAction", static_cast<int>(MA::None)};
  setting[mouseCtrlWheelAction]=        {"mainwindow/manipulate3d/mouseCtrlWheelAction", static_cast<int>(MA::CurserSz)};
  setting[mouseAltWheelAction]=         {"mainwindow/manipulate3d/mouseAltWheelAction", static_cast<int>(MA::None)};
  setting[mouseShiftCtrlWheelAction]=   {"mainwindow/manipulate3d/mouseShiftCtrlWheelAction", static_cast<int>(MA::None)};
  setting[mouseShiftAltWheelAction]=    {"mainwindow/manipulate3d/mouseShiftAltWheelAction", static_cast<int>(MA::None)};
  setting[mouseCtrlAltWheelAction]=     {"mainwindow/manipulate3d/mouseCtrlAltWheelAction", static_cast<int>(MA::None)};
  setting[mouseShiftCtrlAltWheelAction]={"mainwindow/manipulate3d/mouseShiftCtrlAltWheelAction", static_cast<int>(MA::None)};
  using CTA=MyTouchWidget::ClickTapAction;
  setting[mouseNoneLeftClickAction]=        {"mainwindow/manipulate3d/mouseNoneLeftClickAction", static_cast<int>(CTA::SelectTopObject)};
  setting[mouseShiftLeftClickAction]=       {"mainwindow/manipulate3d/mouseShiftLeftClickAction", static_cast<int>(CTA::None)};
  setting[mouseCtrlLeftClickAction]=        {"mainwindow/manipulate3d/mouseCtrlLeftClickAction", static_cast<int>(CTA::ToggleTopObject)};
  setting[mouseAltLeftClickAction]=         {"mainwindow/manipulate3d/mouseAltLeftClickAction", static_cast<int>(CTA::SelectAnyObject)};
  setting[mouseShiftCtrlLeftClickAction]=   {"mainwindow/manipulate3d/mouseShiftCtrlLeftClickAction", static_cast<int>(CTA::None)};
  setting[mouseShiftAltLeftClickAction]=    {"mainwindow/manipulate3d/mouseShiftAltLeftClickAction", static_cast<int>(CTA::None)};
  setting[mouseCtrlAltLeftClickAction]=     {"mainwindow/manipulate3d/mouseCtrlAltLeftClickAction", static_cast<int>(CTA::ToggleAnyObject)};
  setting[mouseShiftCtrlAltLeftClickAction]={"mainwindow/manipulate3d/mouseShiftCtrlAltLeftClickAction", static_cast<int>(CTA::None)};
  setting[mouseNoneRightClickAction]=        {"mainwindow/manipulate3d/mouseNoneRightClickAction", static_cast<int>(CTA::SelectTopObjectAndShowContextMenu)};
  setting[mouseShiftRightClickAction]=       {"mainwindow/manipulate3d/mouseShiftRightClickAction", static_cast<int>(CTA::None)};
  setting[mouseCtrlRightClickAction]=        {"mainwindow/manipulate3d/mouseCtrlRightClickAction", static_cast<int>(CTA::ShowContextMenu)};
  setting[mouseAltRightClickAction]=         {"mainwindow/manipulate3d/mouseAltRightClickAction", static_cast<int>(CTA::ShowContextMenu)};
  setting[mouseShiftCtrlRightClickAction]=   {"mainwindow/manipulate3d/mouseShiftCtrlRightClickAction", static_cast<int>(CTA::None)};
  setting[mouseShiftAltRightClickAction]=    {"mainwindow/manipulate3d/mouseShiftAltRightClickAction", static_cast<int>(CTA::None)};
  setting[mouseCtrlAltRightClickAction]=     {"mainwindow/manipulate3d/mouseCtrlAltRightClickAction", static_cast<int>(CTA::None)};
  setting[mouseShiftCtrlAltRightClickAction]={"mainwindow/manipulate3d/mouseShiftCtrlAltRightClickAction", static_cast<int>(CTA::None)};
  setting[mouseNoneMidClickAction]=        {"mainwindow/manipulate3d/mouseNoneMidClickAction", static_cast<int>(CTA::SeekCameraToPoint)};
  setting[mouseShiftMidClickAction]=       {"mainwindow/manipulate3d/mouseShiftMidClickAction", static_cast<int>(CTA::None)};
  setting[mouseCtrlMidClickAction]=        {"mainwindow/manipulate3d/mouseCtrlMidClickAction", static_cast<int>(CTA::None)};
  setting[mouseAltMidClickAction]=         {"mainwindow/manipulate3d/mouseAltMidClickAction", static_cast<int>(CTA::None)};
  setting[mouseShiftCtrlMidClickAction]=   {"mainwindow/manipulate3d/mouseShiftCtrlMidClickAction", static_cast<int>(CTA::None)};
  setting[mouseShiftAltMidClickAction]=    {"mainwindow/manipulate3d/mouseShiftAltMidClickAction", static_cast<int>(CTA::None)};
  setting[mouseCtrlAltMidClickAction]=     {"mainwindow/manipulate3d/mouseCtrlAltMidClickAction", static_cast<int>(CTA::None)};
  setting[mouseShiftCtrlAltMidClickAction]={"mainwindow/manipulate3d/mouseShiftCtrlAltMidClickAction", static_cast<int>(CTA::None)};
  using CTA=MyTouchWidget::ClickTapAction;
  setting[touchNoneTapAction]=        {"mainwindow/manipulate3d/touchNoneTapAction", static_cast<int>(CTA::SelectTopObject)};
  setting[touchShiftTapAction]=       {"mainwindow/manipulate3d/touchShiftTapAction", static_cast<int>(CTA::None)};
  setting[touchCtrlTapAction]=        {"mainwindow/manipulate3d/touchCtrlTapAction", static_cast<int>(CTA::ToggleTopObject)};
  setting[touchAltTapAction]=         {"mainwindow/manipulate3d/touchAltTapAction", static_cast<int>(CTA::SelectAnyObject)};
  setting[touchShiftCtrlTapAction]=   {"mainwindow/manipulate3d/touchShiftCtrlTapAction", static_cast<int>(CTA::None)};
  setting[touchShiftAltTapAction]=    {"mainwindow/manipulate3d/touchShiftAltTapAction", static_cast<int>(CTA::None)};
  setting[touchCtrlAltTapAction]=     {"mainwindow/manipulate3d/touchCtrlAltTapAction", static_cast<int>(CTA::ToggleAnyObject)};
  setting[touchShiftCtrlAltTapAction]={"mainwindow/manipulate3d/touchShiftCtrlAltTapAction", static_cast<int>(CTA::None)};
  setting[touchNoneLongTapAction]=        {"mainwindow/manipulate3d/touchNoneLongTapAction", static_cast<int>(CTA::SelectTopObjectAndShowContextMenu)};
  setting[touchShiftLongTapAction]=       {"mainwindow/manipulate3d/touchShiftLongTapAction", static_cast<int>(CTA::None)};
  setting[touchCtrlLongTapAction]=        {"mainwindow/manipulate3d/touchCtrlLongTapAction", static_cast<int>(CTA::ShowContextMenu)};
  setting[touchAltLongTapAction]=         {"mainwindow/manipulate3d/touchAltLongTapAction", static_cast<int>(CTA::ShowContextMenu)};
  setting[touchShiftCtrlLongTapAction]=   {"mainwindow/manipulate3d/touchShiftCtrlLongTapAction", static_cast<int>(CTA::None)};
  setting[touchShiftAltLongTapAction]=    {"mainwindow/manipulate3d/touchShiftAltLongTapAction", static_cast<int>(CTA::None)};
  setting[touchCtrlAltLongTapAction]=     {"mainwindow/manipulate3d/touchCtrlAltLongTapAction", static_cast<int>(CTA::None)};
  setting[touchShiftCtrlAltLongTapAction]={"mainwindow/manipulate3d/touchShiftCtrlAltLongTapAction", static_cast<int>(CTA::None)};
  using MA=MyTouchWidget::MoveAction;
  setting[touchNoneMove1Action]=        {"mainwindow/manipulate3d/touchNoneMove1Action", static_cast<int>(MA::RotateAboutSySx)};
  setting[touchShiftMove1Action]=       {"mainwindow/manipulate3d/touchShiftMove1Action", static_cast<int>(MA::None)};
  setting[touchCtrlMove1Action]=        {"mainwindow/manipulate3d/touchCtrlMove1Action", static_cast<int>(MA::None)};
  setting[touchAltMove1Action]=         {"mainwindow/manipulate3d/touchAltMove1Action", static_cast<int>(MA::None)};
  setting[touchShiftCtrlMove1Action]=   {"mainwindow/manipulate3d/touchShiftCtrlMove1Action", static_cast<int>(MA::None)};
  setting[touchShiftAltMove1Action]=    {"mainwindow/manipulate3d/touchShiftAltMove1Action", static_cast<int>(MA::None)};
  setting[touchCtrlAltMove1Action]=     {"mainwindow/manipulate3d/touchCtrlAltMove1Action", static_cast<int>(MA::None)};
  setting[touchShiftCtrlAltMove1Action]={"mainwindow/manipulate3d/touchShiftCtrlAltMove1Action", static_cast<int>(MA::None)};
  setting[touchNoneMove2Action]=        {"mainwindow/manipulate3d/touchNoneMove2Action", static_cast<int>(MA::Translate)};
  setting[touchShiftMove2Action]=       {"mainwindow/manipulate3d/touchShiftMove2Action", static_cast<int>(MA::None)};
  setting[touchCtrlMove2Action]=        {"mainwindow/manipulate3d/touchCtrlMove2Action", static_cast<int>(MA::None)};
  setting[touchAltMove2Action]=         {"mainwindow/manipulate3d/touchAltMove2Action", static_cast<int>(MA::None)};
  setting[touchShiftCtrlMove2Action]=   {"mainwindow/manipulate3d/touchShiftCtrlMove2Action", static_cast<int>(MA::None)};
  setting[touchShiftAltMove2Action]=    {"mainwindow/manipulate3d/touchShiftAltMove2Action", static_cast<int>(MA::None)};
  setting[touchCtrlAltMove2Action]=     {"mainwindow/manipulate3d/touchCtrlAltMove2Action", static_cast<int>(MA::None)};
  setting[touchShiftCtrlAltMove2Action]={"mainwindow/manipulate3d/touchShiftCtrlAltMove2Action", static_cast<int>(MA::None)};
  setting[touchNoneMove2ZoomAction]=        {"mainwindow/manipulate3d/touchNoneMove2ZoomAction", static_cast<int>(MA::Zoom)};
  setting[touchShiftMove2ZoomAction]=       {"mainwindow/manipulate3d/touchShiftMove2ZoomAction", static_cast<int>(MA::None)};
  setting[touchCtrlMove2ZoomAction]=        {"mainwindow/manipulate3d/touchCtrlMove2ZoomAction", static_cast<int>(MA::CameraFocalDistance)};
  setting[touchAltMove2ZoomAction]=         {"mainwindow/manipulate3d/touchAltMove2ZoomAction", static_cast<int>(MA::None)};
  setting[touchShiftCtrlMove2ZoomAction]=   {"mainwindow/manipulate3d/touchShiftCtrlMove2ZoomAction", static_cast<int>(MA::None)};
  setting[touchShiftAltMove2ZoomAction]=    {"mainwindow/manipulate3d/touchShiftAltMove2ZoomAction", static_cast<int>(MA::None)};
  setting[touchCtrlAltMove2ZoomAction]=     {"mainwindow/manipulate3d/touchCtrlAltMove2ZoomAction", static_cast<int>(MA::None)};
  setting[touchShiftCtrlAltMove2ZoomAction]={"mainwindow/manipulate3d/touchShiftCtrlAltMove2ZoomAction", static_cast<int>(MA::None)};
  setting[zoomFacPerPixel]={"mainwindow/manipulate3d/zoomFacPerPixel", 1.005};
  setting[rotAnglePerPixel]={"mainwindow/manipulate3d/rotAnglePerPixel", 0.2};
  setting[relCursorZPerWheel]={"mainwindow/manipulate3d/relCursorZPerWheel", 0.01};
  setting[pickObjectRadius]={"mainwindow/manipulate3d/pickObjectRadius", 3.0};
  setting[inScreenRotateSwitch]={"mainwindow/manipulate3d/inScreenRotateSwitch", 30.0};

  for(auto &[str, value]: setting)
    if(qSettings.contains(str))
      value=qSettings.value(str);
}

AppSettings::~AppSettings() {
  for(auto &[str, value]: setting)
    qSettings.setValue(str, value);
}

std::unique_ptr<AppSettings> appSettings;

class IntSetting : public QWidget {
  public:
    IntSetting(QGridLayout *layout, AppSettings::AS key, const QIcon& icon, const QString &name, const QString &suffix,
               const std::function<void(int)> &set={}, int min=0, int max=numeric_limits<int>::max());
};
IntSetting::IntSetting(QGridLayout *layout, AppSettings::AS key, const QIcon& icon, const QString &name, const QString &suffix,
                       const std::function<void(int)> &set, int min, int max) : QWidget(layout->parentWidget()) {
  QFontInfo fontInfo(font());
  int row=layout->rowCount();
  auto *iconLabel=new QLabel;
  iconLabel->setPixmap(icon.pixmap(fontInfo.pixelSize(),fontInfo.pixelSize()));
  layout->addWidget(iconLabel, row, 0);
  layout->addWidget(new QLabel(name, this), row, 1);
  auto spinBox=new QSpinBox(this);
  spinBox->setMinimum(min);
  spinBox->setMaximum(max);
  spinBox->installEventFilter(&IgnoreWheelEventFilter::instance);
  if(!suffix.isEmpty()) spinBox->setSuffix(" "+suffix);
  layout->addWidget(spinBox, row, 2);
  spinBox->setValue(appSettings->get<int>(key));
  connect(spinBox, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), [key, set](int value){
    appSettings->set(key, value);
    if(set) set(value);
  });
}
class StringSetting : public QWidget {
  public:
    StringSetting(QGridLayout *layout, AppSettings::AS key, const QIcon& icon, const QString &name,
                  const std::function<void(QString)> &set={});
};
StringSetting::StringSetting(QGridLayout *layout, AppSettings::AS key, const QIcon& icon, const QString &name,
                             const std::function<void(QString)> &set) : QWidget(layout->parentWidget()) {
  QFontInfo fontInfo(font());
  int row=layout->rowCount();
  auto *iconLabel=new QLabel;
  iconLabel->setPixmap(icon.pixmap(fontInfo.pixelSize(),fontInfo.pixelSize()));
  layout->addWidget(iconLabel, row, 0);
  layout->addWidget(new QLabel(name, this), row, 1);
  auto textEdit=new QLineEdit(this);
  layout->addWidget(textEdit, row, 2);
  textEdit->setText(appSettings->get<QString>(key));
  connect(textEdit, &QLineEdit::textChanged, [key, set](const QString &value){
    appSettings->set(key, value);
    if(set) set(value);
  });
}
class ChoiceSetting : public QWidget {
  public:
    ChoiceSetting(QGridLayout *layout, AppSettings::AS key, const QIcon& icon, const QString &name,
                  const QStringList &items, const std::function<void(int)> &set={});
};
ChoiceSetting::ChoiceSetting(QGridLayout *layout, AppSettings::AS key, const QIcon& icon, const QString &name,
                             const QStringList &items, const std::function<void(int)> &set) :
                             QWidget(layout->parentWidget()) {
  QFontInfo fontInfo(font());
  int row=layout->rowCount();
  auto *iconLabel=new QLabel;
  iconLabel->setPixmap(icon.pixmap(fontInfo.pixelSize(),fontInfo.pixelSize()));
  layout->addWidget(iconLabel, row, 0);
  layout->addWidget(new QLabel(name, this), row, 1);
  auto comboBox=new QComboBox(this);
  comboBox->addItems(items);
  comboBox->installEventFilter(&IgnoreWheelEventFilter::instance);
  layout->addWidget(comboBox, row, 2);
  comboBox->setCurrentIndex(appSettings->get<int>(key));
  connect(comboBox, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [key, set](int value){
    appSettings->set(key, value);
    if(set) set(value);
  });
}
class DoubleSetting : public QWidget {
  public:
    DoubleSetting(QGridLayout *layout, AppSettings::AS key, const QIcon& icon, const QString &name, const QString &suffix,
                  const std::function<void(double)> &set={}, double min=0, double max=numeric_limits<double>::max(), double step=1);
};
DoubleSetting::DoubleSetting(QGridLayout *layout, AppSettings::AS key, const QIcon& icon, const QString &name, const QString &suffix,
                             const std::function<void(double)> &set, double min, double max, double step) :
                             QWidget(layout->parentWidget()) {
  QFontInfo fontInfo(font());
  int row=layout->rowCount();
  auto *iconLabel=new QLabel;
  iconLabel->setPixmap(icon.pixmap(fontInfo.pixelSize(),fontInfo.pixelSize()));
  layout->addWidget(iconLabel, row, 0);
  layout->addWidget(new QLabel(name, this), row, 1);
  auto doubleSpinBox=new QDoubleSpinBox(this);
  doubleSpinBox->setMinimum(min);
  doubleSpinBox->setMaximum(max);
  doubleSpinBox->setDecimals(6);
  doubleSpinBox->setSingleStep(step);
  doubleSpinBox->installEventFilter(&IgnoreWheelEventFilter::instance);
  if(!suffix.isEmpty()) doubleSpinBox->setSuffix(" "+suffix);
  layout->addWidget(doubleSpinBox, row, 2);
  doubleSpinBox->setValue(appSettings->get<double>(key));
  connect(doubleSpinBox, static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), [key, set](double value){
    appSettings->set(key, value);
    if(set) set(value);
  });
}
class ColorSetting : public QWidget {
  public:
    ColorSetting(QGridLayout *layout, AppSettings::AS key, const QIcon& icon, const QString &name, const std::function<void(const QColor&)> &set={});
};
ColorSetting::ColorSetting(QGridLayout *layout, AppSettings::AS key, const QIcon& icon, const QString &name,
                           const std::function<void(const QColor&)> &set) : QWidget(layout->parentWidget()) {
  QFontInfo fontInfo(font());
  int row=layout->rowCount();
  auto *iconLabel=new QLabel;
  iconLabel->setPixmap(icon.pixmap(fontInfo.pixelSize(),fontInfo.pixelSize()));
  layout->addWidget(iconLabel, row, 0);
  layout->addWidget(new QLabel(name, this), row, 1);
  auto colorDialog=new QColorDialog(this);
  colorDialog->setWindowTitle("Select Color: "+name);
  colorDialog->setOption(QColorDialog::NoButtons);
  colorDialog->setCurrentColor(appSettings->get<QColor>(key));
  connect(colorDialog, &QColorDialog::currentColorChanged, [key, set](const QColor& color){
    appSettings->set(key, color);
    if(set) set(color);
  });
  auto *showDL=new QPushButton("Color...");
  connect(showDL, &QPushButton::clicked, this, [colorDialog](){ colorDialog->show(); });
  layout->addWidget(showDL, row, 2);
}

SettingsDialog::SettingsDialog(QWidget *parent) : QDialog(parent) {
  setWindowTitle("Settings");
  setMinimumSize(500, 500);
  auto mainLayout=new QGridLayout;
  mainLayout->setColumnStretch(0, 0);
  mainLayout->setColumnStretch(1, 1);
  setLayout(mainLayout);
  // display settings icon and label
  auto *settingsTitle=new QLabel("Settings");
  auto font=settingsTitle->font();
  font.setBold(true);
  font.setPointSize(1.5*font.pointSize());
  settingsTitle->setFont(font);
  QFontInfo fontInfo(font);
  auto *settingsIcon=new QLabel;
  settingsIcon->setPixmap(Utils::QIconCached("settings.svg").pixmap(fontInfo.pixelSize(),fontInfo.pixelSize()));
  mainLayout->addWidget(settingsIcon, 0, 0);
  mainLayout->addWidget(settingsTitle, 0, 1);
  auto layout=new QGridLayout;
  layout->setColumnStretch(0, 0);
  layout->setColumnStretch(1, 0);
  layout->setColumnStretch(2, 1);
  auto scrollWidget=new QWidget;
  scrollWidget->setLayout(layout);
  auto scrollArea=new QScrollArea;
  scrollArea->setWidgetResizable(true);
  Utils::enableTouch(scrollArea);
  mainLayout->addWidget(scrollArea, 1, 0, 1, 2);

  new IntSetting(layout, AppSettings::hdf5RefreshDelta, Utils::QIconCached("time.svg"), "HDF5 refresh delta:", "ms", [](int value){
    MainWindow::getInstance()->setHDF5RefreshDelta(value);
  });
  new ChoiceSetting(layout, AppSettings::cameraType, Utils::QIconCached("camera.svg"), "Camera type:",
                    {"Orthographic", "Perspective"}, [](int value){
    MainWindow::getInstance()->setCameraType(value==0 ? SoOrthographicCamera::getClassTypeId() : SoPerspectiveCamera::getClassTypeId());
  });
  new ChoiceSetting(layout, AppSettings::stereoType, Utils::QIconCached(""), "Stereo view type:",
                    {"None", "Left/Right"}, [](int value){
    MainWindow::getInstance()->reinit3DView(static_cast<MainWindow::StereoType>(value));
  });
  new DoubleSetting(layout, AppSettings::stereoOffset, Utils::QIconCached(""), "Stereo view eye distance:", "m", [](double value){
    MainWindow::getInstance()->setStereoOffset(value);
  }, 0, numeric_limits<double>::max(), 0.01);
  new IntSetting(layout, AppSettings::tapAndHoldTimeout, Utils::QIconCached("time.svg"), "Tap and hold timeout:", "ms", [](int value){
    QTapAndHoldGesture::setTimeout(value);
  });
  new IntSetting(layout, AppSettings::shortAniTime, Utils::QIconCached("time.svg"), "Short animation time:", "ms");
  new DoubleSetting(layout, AppSettings::zoomFacPerPixel, Utils::QIconCached("zoom.svg"), "Zoom factor:", "1/px", [](double value){
    MainWindow::getInstance()->glViewerWG->setZoomFacPerPixel(value);
  }, 0, numeric_limits<double>::max(), 0.0001);
  new DoubleSetting(layout, AppSettings::rotAnglePerPixel, Utils::QIconCached("angle.svg"), "Rotation angle:", "deg/px", [](double value){
    MainWindow::getInstance()->glViewerWG->setRotAnglePerPixel(value);
  }, numeric_limits<double>::min(), numeric_limits<double>::max(), 0.01);
  new DoubleSetting(layout, AppSettings::relCursorZPerWheel, Utils::QIconCached("angle.svg"), "Relative cursor-z change per wheel:", "", [](double value){
    MainWindow::getInstance()->glViewerWG->setRelCursorZPerWheel(value);
  }, 0, 1, 0.001);
  new DoubleSetting(layout, AppSettings::pickObjectRadius, Utils::QIconCached("target.svg"), "Pick object radius:", "px", [](double value){
    MainWindow::getInstance()->glViewerWG->setPickObjectRadius(value);
  }, 0, numeric_limits<double>::max(), 0.1);
  new DoubleSetting(layout, AppSettings::inScreenRotateSwitch, Utils::QIconCached("angle.svg"), "In screen rotate barrier:", "deg", [](double value){
    MainWindow::getInstance()->glViewerWG->setInScreenRotateSwitch(value);
  });
  new DoubleSetting(layout, AppSettings::anglePerKeyPress, Utils::QIconCached("angle.svg"), "Rotation per keypress:", "deg", {},
                    numeric_limits<double>::min(), numeric_limits<double>::max(), 0.5);
  new DoubleSetting(layout, AppSettings::speedChangeFactor, Utils::QIconCached("speed.svg"), "Animation speed factor:", "1/key", {},
                    0, numeric_limits<double>::max(), 0.01);
  new DoubleSetting(layout, AppSettings::mouseCursorSize, Utils::QIconCached("mouse.svg"), "Mouse cursor size:", "%", [](double value){
    MainWindow::getInstance()->mouseCursorSizeField.setValue(value);
  }, 0, 100, 1);
  #define MOUSECLICK(mod, button) \
    new ChoiceSetting(layout, AppSettings::mouse##mod##button##ClickAction, Utils::QIconCached("mouse.svg"), \
        ("Mouse "+boost::algorithm::to_lower_copy(string(#button))+" click ("+#mod+"-Mod):").c_str(), { \
        "No action", \
        "Select top object", \
        "Toggle top object", \
        "Select any object", \
        "Toggle any object", \
        "Show context menu", \
        "Select top object and show context menu", \
        "Select any object and show context menu", \
        "Seek camera to point", \
      }, [](int value){ \
      MainWindow::getInstance()->glViewerWG->setMouse##button##ClickAction( \
        MyTouchWidget::Modifier::mod, static_cast<MyTouchWidget::ClickTapAction>(value)); \
    })
  MOUSECLICK(None        , Left);
  MOUSECLICK(Shift       , Left);
  MOUSECLICK(Ctrl        , Left);
  MOUSECLICK(Alt         , Left);
  MOUSECLICK(ShiftCtrl   , Left);
  MOUSECLICK(ShiftAlt    , Left);
  MOUSECLICK(CtrlAlt     , Left);
  MOUSECLICK(ShiftCtrlAlt, Left);
  MOUSECLICK(None        , Right);
  MOUSECLICK(Shift       , Right);
  MOUSECLICK(Ctrl        , Right);
  MOUSECLICK(Alt         , Right);
  MOUSECLICK(ShiftCtrl   , Right);
  MOUSECLICK(ShiftAlt    , Right);
  MOUSECLICK(CtrlAlt     , Right);
  MOUSECLICK(ShiftCtrlAlt, Right);
  MOUSECLICK(None        , Mid);
  MOUSECLICK(Shift       , Mid);
  MOUSECLICK(Ctrl        , Mid);
  MOUSECLICK(Alt         , Mid);
  MOUSECLICK(ShiftCtrl   , Mid);
  MOUSECLICK(ShiftAlt    , Mid);
  MOUSECLICK(CtrlAlt     , Mid);
  MOUSECLICK(ShiftCtrlAlt, Mid);
  #define MOUSEMOVE(mod, button) \
    new ChoiceSetting(layout, AppSettings::mouse##mod##button##MoveAction, Utils::QIconCached("mouse.svg"), \
        ("Mouse "+boost::algorithm::to_lower_copy(string(#button))+" move ("+#mod+"-Mod):").c_str(), { \
        "No action", \
        "Change frame", \
        "Zoom", \
        "Change camera focal distance", \
        "Change cursor screen-z position", \
        "Rotate about screen-z", \
        "Translate", \
        "Rotate about screen-y,screen-x", \
        "Rotate about world-x,screen-x", \
        "Rotate about world-y,screen-x", \
        "Rotate about world-z,screen-x", \
      }, [](int value){ \
      MainWindow::getInstance()->glViewerWG->setMouse##button##MoveAction( \
        MyTouchWidget::Modifier::mod, static_cast<MyTouchWidget::MoveAction>(value)); \
    })
  MOUSEMOVE(None        , Left);
  MOUSEMOVE(Shift       , Left);
  MOUSEMOVE(Ctrl        , Left);
  MOUSEMOVE(Alt         , Left);
  MOUSEMOVE(ShiftCtrl   , Left);
  MOUSEMOVE(ShiftAlt    , Left);
  MOUSEMOVE(CtrlAlt     , Left);
  MOUSEMOVE(ShiftCtrlAlt, Left);
  MOUSEMOVE(None        , Right);
  MOUSEMOVE(Shift       , Right);
  MOUSEMOVE(Ctrl        , Right);
  MOUSEMOVE(Alt         , Right);
  MOUSEMOVE(ShiftCtrl   , Right);
  MOUSEMOVE(ShiftAlt    , Right);
  MOUSEMOVE(CtrlAlt     , Right);
  MOUSEMOVE(ShiftCtrlAlt, Right);
  MOUSEMOVE(None        , Mid);
  MOUSEMOVE(Shift       , Mid);
  MOUSEMOVE(Ctrl        , Mid);
  MOUSEMOVE(Alt         , Mid);
  MOUSEMOVE(ShiftCtrl   , Mid);
  MOUSEMOVE(ShiftAlt    , Mid);
  MOUSEMOVE(CtrlAlt     , Mid);
  MOUSEMOVE(ShiftCtrlAlt, Mid);
  #define MOUSEWHEEL(mod) \
    new ChoiceSetting(layout, AppSettings::mouse##mod##WheelAction, Utils::QIconCached("mouse.svg"), \
        ("Mouse wheel ("+string(#mod)+"-Mod):").c_str(), { \
        "No action", \
        "Change frame", \
        "Zoom", \
        "Change camera focal distance", \
        "Change cursor screen-z position", \
        "Rotate about screen-z", \
        "<not available>", \
        "<not available>", \
        "<not available>", \
        "<not available>", \
        "<not available>", \
      }, [](int value){ \
      MainWindow::getInstance()->glViewerWG->setMouseWheelAction( \
        MyTouchWidget::Modifier::mod, static_cast<MyTouchWidget::MoveAction>(value)); \
    })
  MOUSEWHEEL(None        );
  MOUSEWHEEL(Shift       );
  MOUSEWHEEL(Ctrl        );
  MOUSEWHEEL(Alt         );
  MOUSEWHEEL(ShiftCtrl   );
  MOUSEWHEEL(ShiftAlt    );
  MOUSEWHEEL(CtrlAlt     );
  MOUSEWHEEL(ShiftCtrlAlt);
  #define TOUCHTAP(mod, tap) \
    new ChoiceSetting(layout, AppSettings::touch##mod##tap##Action, Utils::QIconCached("touch.svg"), \
        ("Touch "+boost::algorithm::to_lower_copy(string(#tap))+" ("+#mod+"-Mod):").c_str(), { \
        "No action", \
        "Select top object", \
        "Toggle top object", \
        "Select any object", \
        "Toggle any object", \
        "Show context menu", \
        "Select top object and show context menu", \
        "Select any object and show context menu", \
        "Seek camera to point", \
      }, [](int value){ \
      MainWindow::getInstance()->glViewerWG->setTouch##tap##Action( \
        MyTouchWidget::Modifier::mod, static_cast<MyTouchWidget::ClickTapAction>(value)); \
    })
  TOUCHTAP(None        , Tap);
  TOUCHTAP(Shift       , Tap);
  TOUCHTAP(Ctrl        , Tap);
  TOUCHTAP(Alt         , Tap);
  TOUCHTAP(ShiftCtrl   , Tap);
  TOUCHTAP(ShiftAlt    , Tap);
  TOUCHTAP(CtrlAlt     , Tap);
  TOUCHTAP(ShiftCtrlAlt, Tap);
  TOUCHTAP(None        , LongTap);
  TOUCHTAP(Shift       , LongTap);
  TOUCHTAP(Ctrl        , LongTap);
  TOUCHTAP(Alt         , LongTap);
  TOUCHTAP(ShiftCtrl   , LongTap);
  TOUCHTAP(ShiftAlt    , LongTap);
  TOUCHTAP(CtrlAlt     , LongTap);
  TOUCHTAP(ShiftCtrlAlt, LongTap);
  #define TOUCHMOVE(mod, tap) \
    new ChoiceSetting(layout, AppSettings::touch##mod##tap##Action, Utils::QIconCached("touch.svg"), \
        ("Touch "+boost::algorithm::to_lower_copy(string(#tap))+" finger ("+#mod+"-Mod):").c_str(), { \
        "No action", \
        "Change frame", \
        "<not available>", \
        "Change camera focal distance", \
        "Change cursor screen-z position", \
        "<not available>", \
        "Translate", \
        "Rotate about screen-y,screen-x", \
        "Rotate about world-x,screen-x", \
        "Rotate about world-y,screen-x", \
        "Rotate about world-z,screen-x", \
      }, [](int value){ \
      MainWindow::getInstance()->glViewerWG->setTouch##tap##Action( \
        MyTouchWidget::Modifier::mod, static_cast<MyTouchWidget::MoveAction>(value)); \
    })
  TOUCHMOVE(None        , Move1);
  TOUCHMOVE(Shift       , Move1);
  TOUCHMOVE(Ctrl        , Move1);
  TOUCHMOVE(Alt         , Move1);
  TOUCHMOVE(ShiftCtrl   , Move1);
  TOUCHMOVE(ShiftAlt    , Move1);
  TOUCHMOVE(CtrlAlt     , Move1);
  TOUCHMOVE(ShiftCtrlAlt, Move1);
  TOUCHMOVE(None        , Move2);
  TOUCHMOVE(Shift       , Move2);
  TOUCHMOVE(Ctrl        , Move2);
  TOUCHMOVE(Alt         , Move2);
  TOUCHMOVE(ShiftCtrl   , Move2);
  TOUCHMOVE(ShiftAlt    , Move2);
  TOUCHMOVE(CtrlAlt     , Move2);
  TOUCHMOVE(ShiftCtrlAlt, Move2);
  #define TOUCHMOVEZOOM(mod) \
    new ChoiceSetting(layout, AppSettings::touch##mod##Move2ZoomAction, Utils::QIconCached("touch.svg"), \
        ("Touch zoom ("+string(#mod)+"-Mod):").c_str(), { \
        "No action", \
        "<not available>", \
        "Zoom", \
        "Change camera focal distance", \
        "<not available>", \
        "<not available>", \
        "<not available>", \
        "<not available>", \
        "<not available>", \
        "<not available>", \
        "<not available>", \
      }, [](int value){ \
      MainWindow::getInstance()->glViewerWG->setTouch##Move2ZoomAction( \
        MyTouchWidget::Modifier::mod, static_cast<MyTouchWidget::MoveAction>(value)); \
    })
  TOUCHMOVEZOOM(None        );
  TOUCHMOVEZOOM(Shift       );
  TOUCHMOVEZOOM(Ctrl        );
  TOUCHMOVEZOOM(Alt         );
  TOUCHMOVEZOOM(ShiftCtrl   );
  TOUCHMOVEZOOM(ShiftAlt    );
  TOUCHMOVEZOOM(CtrlAlt     );
  TOUCHMOVEZOOM(ShiftCtrlAlt);
  new DoubleSetting(layout, AppSettings::outlineShilouetteEdgeLineWidth, Utils::QIconCached("olselinewidth.svg"), "Outline line width:", "px", [](double value){
    MainWindow::getInstance()->olseDrawStyle->lineWidth.setValue(value);
  }, 0, numeric_limits<double>::max(), 0.1);
  new ColorSetting(layout, AppSettings::outlineShilouetteEdgeLineColor, Utils::QIconCached("olsecolor.svg"), "Outline line color:", [](const QColor &value){
    auto rgb=value.rgb();
    MainWindow::getInstance()->olseColor->rgb.set1Value(0, qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
  });
  new DoubleSetting(layout, AppSettings::boundingBoxLineWidth, Utils::QIconCached("lines.svg"), "Bounding box line width:", "px", [](double value){
    MainWindow::getInstance()->bboxDrawStyle->lineWidth.setValue(value);
  }, 0, numeric_limits<double>::max(), 0.1);
  new ColorSetting(layout, AppSettings::boundingBoxLineColor, Utils::QIconCached("lines.svg"), "Bounding box line color:", [](const QColor &value){
    auto rgb=value.rgb();
    MainWindow::getInstance()->bboxColor->rgb.set1Value(0, qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
  });
  new DoubleSetting(layout, AppSettings::highlightLineWidth, Utils::QIconCached("lines.svg"), "Highlight line width:", "px", [](double value){
    MainWindow::getInstance()->highlightDrawStyle->lineWidth.setValue(value);
  }, 0, numeric_limits<double>::max(), 0.1);
  new ColorSetting(layout, AppSettings::highlightLineColor, Utils::QIconCached("lines.svg"), "Highlight line color:", [](const QColor &value){
    auto rgb=value.rgb();
    MainWindow::getInstance()->highlightColor->rgb.set1Value(0, qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
  });
  new ChoiceSetting(layout, AppSettings::complexityType, Utils::QIconCached("complexitytype.svg"), "Complxity type:",
                    {"Object space", "Screen space", "Bounding box"}, [](int value){
    MainWindow::getInstance()->complexity->type.setValue( value==0 ? SoComplexity::OBJECT_SPACE :
                                                         (value==1 ? SoComplexity::SCREEN_SPACE :
                                                                     SoComplexity::BOUNDING_BOX));
  });
  new DoubleSetting(layout, AppSettings::complexityValue, Utils::QIconCached("complexityvalue.svg"), "Complexity value:", "", [](double value){
    MainWindow::getInstance()->complexity->value.setValue(value);
  }, 0, 1, 0.1);
  new ColorSetting(layout, AppSettings::topBackgroudColor, Utils::QIconCached("bgcolor.svg"), "Top background color:", [](const QColor &color){
    auto rgb=color.rgb();
    MainWindow::getInstance()->bgColor->set1Value(2, qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
    MainWindow::getInstance()->bgColor->set1Value(3, qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
    MainWindow::getInstance()->fgColorTop->set1Value(0, 1-(color.value()+127)/255,1-(color.value()+127)/255,1-(color.value()+127)/255);
    MainWindow::getInstance()->updateScene();
  });
  new ColorSetting(layout, AppSettings::bottomBackgroundColor, Utils::QIconCached("bgcolor.svg"), "Bottom background color:", [](const QColor &color){
    auto rgb=color.rgb();
    MainWindow::getInstance()->bgColor->set1Value(0, qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
    MainWindow::getInstance()->bgColor->set1Value(1, qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
    MainWindow::getInstance()->fgColorBottom->set1Value(0, 1-(color.value()+127)/255,1-(color.value()+127)/255,1-(color.value()+127)/255);
    MainWindow::getInstance()->updateScene();
  });

  scrollArea->setWidget(scrollWidget);
}
void SettingsDialog::closeEvent(QCloseEvent *event) {
  appSettings->set(AppSettings::settingsDialog_geometry, saveGeometry());
  QDialog::closeEvent(event);
}
void SettingsDialog::showEvent(QShowEvent *event) {
  restoreGeometry(appSettings->get<QByteArray>(AppSettings::settingsDialog_geometry));
  QDialog::showEvent(event);
}

}
