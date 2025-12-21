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
#include <Inventor/actions/SoSearchAction.h>
#include "SoSpecial.h"
#include "mainwindow.h"
#include "mytouchwidget.h"
#include "mbxmlutilshelper/last_write_time.h"
#include <iostream>
#include <QScroller>
#include <QDialog>
#include <QTabWidget>
#include <QPushButton>
#include <QLineEdit>
#include <QMessageBox>
#include <QGridLayout>
#include <QComboBox>
#include <QLabel>
#include <QBitArray>
#include <boost/dll.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/container_hash/hash.hpp>

using namespace std;

namespace OpenMBVGUI {

unordered_map<size_t, Utils::SoDeleteSeparator> Utils::ivCache;
unordered_map<string, QIcon> Utils::iconCache;
bool Utils::initialized=false;

void Utils::initialize() {
  if(initialized) return;
  initialized=true;

  // tess
  gluTessCallback(tess(), GLU_TESS_BEGIN_DATA, (void (CALLMETHOD *)())tessBeginCB);
  gluTessCallback(tess(), GLU_TESS_VERTEX, (void (CALLMETHOD *)())tessVertexCB);
  gluTessCallback(tess(), GLU_TESS_END, (void (CALLMETHOD *)())tessEndCB);
}

void Utils::deinitialize() {
  if(!initialized) return;
  initialized=false;

  iconCache.clear();
}

const QIcon& Utils::QIconCached(const string &basefilename) {
  pair<unordered_map<string, QIcon>::iterator, bool> ins=iconCache.insert(pair<string, QIcon>(basefilename, QIcon()));
  if(ins.second) {
    static const string iconPath((boost::dll::program_location().parent_path().parent_path()/"share"/"openmbv"/"icons").string()+"/");
    return ins.first->second=QIcon((iconPath+basefilename).c_str());
  }
  return ins.first->second;
}

SoSeparator* Utils::SoDBreadAllFileNameCached(const string &filename, size_t hash) {
  boost::filesystem::path fn(filename);
  if(!boost::filesystem::exists(fn)) {
    QString str("IV file %1 does not exist."); str=str.arg(filename.c_str());
    MainWindow::getInstance()->statusBar()->showMessage(str);
    msgStatic(Warn)<<str.toStdString()<<endl;
    return new SoSeparator;
  }
  size_t fullHash = boost::hash<pair<string, size_t>>{}(make_pair(boost::filesystem::canonical(fn).string(), hash));
  auto [it, created]=ivCache.emplace(fullHash, SoDeleteSeparator());
  auto newFileTime = boost::myfilesystem::last_write_time(filename);
  if(created || newFileTime > it->second.fileTime) {
    it->second.fileTime = newFileTime;
    SoInput in;
    if(in.openFile(filename.c_str(), true)) { // if file can be opened, read it
      it->second.sep.reset(SoDB::readAll(&in)); // stored in a global cache => false positive in valgrind
      if(it->second.sep)
        return it->second.sep.get();
    }
    // error case
    QString str("Failed to load IV file %1."); str=str.arg(filename.c_str());
    MainWindow::getInstance()->statusBar()->showMessage(str);
    msgStatic(Warn)<<str.toStdString()<<endl;
    ivCache.erase(it);
    return nullptr;
  }
  return it->second.sep.get();
}

SoSeparator* Utils::SoDBreadAllContentCached(const string &content, size_t hash) {
  size_t fullHash = boost::hash<pair<string, size_t>>{}(make_pair(content, hash));
  auto [it, created]=ivCache.emplace(fullHash, SoDeleteSeparator());
  if(created) {
    SoInput in;
    in.setBuffer(content.data(), content.size());
    it->second.sep.reset(SoDB::readAll(&in)); // stored in a global cache => false positive in valgrind
    if(it->second.sep)
      return it->second.sep.get();
    // error case
    QString str("Failed to load IV content from string.");
    MainWindow::getInstance()->statusBar()->showMessage(str);
    msgStatic(Warn)<<str.toStdString()<<endl;
    ivCache.erase(it);
    return nullptr;
  }
  return it->second.sep.get();
}

SoNode* Utils::getChildNodeByName(SoGroup *sep, const SbName &name) {
  // get the node by name
  auto node = SoNode::getByName(name);
  // if not found return nullptr
  if(!node)
    return nullptr;
  // check if this node is a child of sep
  SoSearchAction sa;
  sa.setNode(node);
  sa.apply(sep);
  // if so, return the node
  if(sa.getPath()!=nullptr)
    return node;
  // if not return nullptr
  return nullptr;
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
GLUtesselator *Utils::tess() {
  auto d=[](GLUtesselator* x) {
    gluDeleteTess(x);
  };
  static unique_ptr<GLUtesselator, decltype(d)> t(gluNewTess(), d);
  return t.get();
}
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
  setting[stereoAspectRatio]={"mainwindow/sceneGraph/stereoAspectRatio", 2};
  setting[tapAndHoldTimeout]={"mainwindow/manipulate3d/tapAndHoldTimeout", 700};
  setting[outlineShilouetteEdgeLineWidth]={"mainwindow/sceneGraph/outlineShilouetteEdgeLineWidth", 1.0};
  setting[outlineShilouetteEdgeLineColor]={"mainwindow/sceneGraph/outlineShilouetteEdgeLineColor", QColor(0,0,0)};
  setting[boundingBoxLineWidth]={"mainwindow/sceneGraph/boundingBoxLineWidth", 2.0};
  setting[boundingBoxLineColor]={"mainwindow/sceneGraph/boundingBoxLineColor", QColor(0,190,0)};
  setting[highlightMethod]={"mainwindow/sceneGraph/highlightMethod", 0};
  setting[highlightTransparencyFactor]={"mainwindow/sceneGraph/highlightTransparencyFactor", 0.8};
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
  setting[exportdialog_bitrate]={"exportdialog/kbps", 10000}; // bitrate is a old key which should no longer be used
  setting[exportdialog_videocmd]={"exportdialog/videocommand", // videocmd is a old key which should no longer be used
    "ffmpeg -framerate %F -i %I -c:v libvpx-vp9 -b:v %B -pass 1 -f null /dev/null 2>&1 && "
    "ffmpeg -framerate %F -i %I -c:v libvpx-vp9 -b:v %B -pass 2 %O 2>&1"};
  setting[exportdialog_videoext]={"exportdialog/videoextension", "webm"};
  setting[propertydialog_geometry]={"propertydialog/geometry", QVariant()};
  setting[dialogstereo_geometry]={"dialogstereo/geometry", QVariant()};
  setting[mouseCursor3D]={"mainwindow/manipulate3d/mouseCursor3D", true};
  setting[mouseCursorSize]={"mainwindow/manipulate3d/mouseCursorSize", 5.0};
  using MA=MyTouchWidget::MoveAction;
  setting[mouseNoneLeftMoveAction]=        {"mainwindow/manipulate3d/mouseNoneLeftMoveAction", static_cast<int>(MA::RotateAboutSySx)};
  setting[mouseShiftLeftMoveAction]=       {"mainwindow/manipulate3d/mouseShiftLeftMoveAction", static_cast<int>(MA::Zoom)};
  setting[mouseCtrlLeftMoveAction]=        {"mainwindow/manipulate3d/mouseCtrlLeftMoveAction", static_cast<int>(MA::RotateAboutSz)};
  setting[mouseAltLeftMoveAction]=         {"mainwindow/manipulate3d/mouseAltLeftMoveAction", static_cast<int>(MA::ChangeFrame)};
  setting[mouseShiftCtrlLeftMoveAction]=   {"mainwindow/manipulate3d/mouseShiftCtrlLeftMoveAction", static_cast<int>(MA::CameraAngle)};
  setting[mouseShiftAltLeftMoveAction]=    {"mainwindow/manipulate3d/mouseShiftAltLeftMoveAction", static_cast<int>(MA::CameraAndRotationPointSz)};
  setting[mouseCtrlAltLeftMoveAction]=     {"mainwindow/manipulate3d/mouseCtrlAltLeftMoveAction", static_cast<int>(MA::None)};
  setting[mouseShiftCtrlAltLeftMoveAction]={"mainwindow/manipulate3d/mouseShiftCtrlAltLeftMoveAction", static_cast<int>(MA::None)};
  setting[mouseNoneRightMoveAction]=        {"mainwindow/manipulate3d/mouseNoneRightMoveAction", static_cast<int>(MA::Translate)};
  setting[mouseShiftRightMoveAction]=       {"mainwindow/manipulate3d/mouseShiftRightMoveAction", static_cast<int>(MA::None)};
  setting[mouseCtrlRightMoveAction]=        {"mainwindow/manipulate3d/mouseCtrlRightMoveAction", static_cast<int>(MA::CursorSz)};
  setting[mouseAltRightMoveAction]=         {"mainwindow/manipulate3d/mouseAltRightMoveAction", static_cast<int>(MA::None)};
  setting[mouseShiftCtrlRightMoveAction]=   {"mainwindow/manipulate3d/mouseShiftCtrlRightMoveAction", static_cast<int>(MA::None)};
  setting[mouseShiftAltRightMoveAction]=    {"mainwindow/manipulate3d/mouseShiftAltRightMoveAction", static_cast<int>(MA::None)};
  setting[mouseCtrlAltRightMoveAction]=     {"mainwindow/manipulate3d/mouseCtrlAltRightMoveAction", static_cast<int>(MA::None)};
  setting[mouseShiftCtrlAltRightMoveAction]={"mainwindow/manipulate3d/mouseShiftCtrlAltRightMoveAction", static_cast<int>(MA::None)};
  setting[mouseNoneMidMoveAction]=        {"mainwindow/manipulate3d/mouseNoneMidMoveAction", static_cast<int>(MA::Zoom)};
  setting[mouseShiftMidMoveAction]=       {"mainwindow/manipulate3d/mouseShiftMidMoveAction", static_cast<int>(MA::CameraNearPlane)};
  setting[mouseCtrlMidMoveAction]=        {"mainwindow/manipulate3d/mouseCtrlMidMoveAction", static_cast<int>(MA::CameraAngle)};
  setting[mouseAltMidMoveAction]=         {"mainwindow/manipulate3d/mouseAltMidMoveAction", static_cast<int>(MA::CameraAndRotationPointSz)};
  setting[mouseShiftCtrlMidMoveAction]=   {"mainwindow/manipulate3d/mouseShiftCtrlMidMoveAction", static_cast<int>(MA::None)};
  setting[mouseShiftAltMidMoveAction]=    {"mainwindow/manipulate3d/mouseShiftAltMidMoveAction", static_cast<int>(MA::None)};
  setting[mouseCtrlAltMidMoveAction]=     {"mainwindow/manipulate3d/mouseCtrlAltMidMoveAction", static_cast<int>(MA::None)};
  setting[mouseShiftCtrlAltMidMoveAction]={"mainwindow/manipulate3d/mouseShiftCtrlAltMidMoveAction", static_cast<int>(MA::None)};
  setting[mouseNoneWheelAction]=        {"mainwindow/manipulate3d/mouseNoneWheelAction", static_cast<int>(MA::ChangeFrame)};
  setting[mouseShiftWheelAction]=       {"mainwindow/manipulate3d/mouseShiftWheelAction", static_cast<int>(MA::Zoom)};
  setting[mouseCtrlWheelAction]=        {"mainwindow/manipulate3d/mouseCtrlWheelAction", static_cast<int>(MA::CursorSz)};
  setting[mouseAltWheelAction]=         {"mainwindow/manipulate3d/mouseAltWheelAction", static_cast<int>(MA::RotateAboutSz)};
  setting[mouseShiftCtrlWheelAction]=   {"mainwindow/manipulate3d/mouseShiftCtrlWheelAction", static_cast<int>(MA::None)};
  setting[mouseShiftAltWheelAction]=    {"mainwindow/manipulate3d/mouseShiftAltWheelAction", static_cast<int>(MA::None)};
  setting[mouseCtrlAltWheelAction]=     {"mainwindow/manipulate3d/mouseCtrlAltWheelAction", static_cast<int>(MA::None)};
  setting[mouseShiftCtrlAltWheelAction]={"mainwindow/manipulate3d/mouseShiftCtrlAltWheelAction", static_cast<int>(MA::None)};
  using CTA=MyTouchWidget::ClickTapAction;
  setting[mouseNoneLeftClickAction]=        {"mainwindow/manipulate3d/mouseNoneLeftClickAction", static_cast<int>(CTA::SelectTopObject)};
  setting[mouseShiftLeftClickAction]=       {"mainwindow/manipulate3d/mouseShiftLeftClickAction", static_cast<int>(CTA::SetRotationPointAndCursorSz)};
  setting[mouseCtrlLeftClickAction]=        {"mainwindow/manipulate3d/mouseCtrlLeftClickAction", static_cast<int>(CTA::ToggleTopObject)};
  setting[mouseAltLeftClickAction]=         {"mainwindow/manipulate3d/mouseAltLeftClickAction", static_cast<int>(CTA::SelectAnyObject)};
  setting[mouseShiftCtrlLeftClickAction]=   {"mainwindow/manipulate3d/mouseShiftCtrlLeftClickAction", static_cast<int>(CTA::None)};
  setting[mouseShiftAltLeftClickAction]=    {"mainwindow/manipulate3d/mouseShiftAltLeftClickAction", static_cast<int>(CTA::None)};
  setting[mouseCtrlAltLeftClickAction]=     {"mainwindow/manipulate3d/mouseCtrlAltLeftClickAction", static_cast<int>(CTA::ToggleAnyObject)};
  setting[mouseShiftCtrlAltLeftClickAction]={"mainwindow/manipulate3d/mouseShiftCtrlAltLeftClickAction", static_cast<int>(CTA::None)};
  setting[mouseNoneRightClickAction]=        {"mainwindow/manipulate3d/mouseNoneRightClickAction", static_cast<int>(CTA::SelectTopObjectAndShowContextMenu)};
  setting[mouseShiftRightClickAction]=       {"mainwindow/manipulate3d/mouseShiftRightClickAction", static_cast<int>(CTA::None)};
  setting[mouseCtrlRightClickAction]=        {"mainwindow/manipulate3d/mouseCtrlRightClickAction", static_cast<int>(CTA::ShowContextMenu)};
  setting[mouseAltRightClickAction]=         {"mainwindow/manipulate3d/mouseAltRightClickAction", static_cast<int>(CTA::SelectAnyObjectAndShowContextMenu)};
  setting[mouseShiftCtrlRightClickAction]=   {"mainwindow/manipulate3d/mouseShiftCtrlRightClickAction", static_cast<int>(CTA::None)};
  setting[mouseShiftAltRightClickAction]=    {"mainwindow/manipulate3d/mouseShiftAltRightClickAction", static_cast<int>(CTA::None)};
  setting[mouseCtrlAltRightClickAction]=     {"mainwindow/manipulate3d/mouseCtrlAltRightClickAction", static_cast<int>(CTA::None)};
  setting[mouseShiftCtrlAltRightClickAction]={"mainwindow/manipulate3d/mouseShiftCtrlAltRightClickAction", static_cast<int>(CTA::None)};
  setting[mouseNoneMidClickAction]=        {"mainwindow/manipulate3d/mouseNoneMidClickAction", static_cast<int>(CTA::SetRotationPointAndCursorSz)};
  setting[mouseShiftMidClickAction]=       {"mainwindow/manipulate3d/mouseShiftMidClickAction", static_cast<int>(CTA::None)};
  setting[mouseCtrlMidClickAction]=        {"mainwindow/manipulate3d/mouseCtrlMidClickAction", static_cast<int>(CTA::None)};
  setting[mouseAltMidClickAction]=         {"mainwindow/manipulate3d/mouseAltMidClickAction", static_cast<int>(CTA::None)};
  setting[mouseShiftCtrlMidClickAction]=   {"mainwindow/manipulate3d/mouseShiftCtrlMidClickAction", static_cast<int>(CTA::None)};
  setting[mouseShiftAltMidClickAction]=    {"mainwindow/manipulate3d/mouseShiftAltMidClickAction", static_cast<int>(CTA::None)};
  setting[mouseCtrlAltMidClickAction]=     {"mainwindow/manipulate3d/mouseCtrlAltMidClickAction", static_cast<int>(CTA::None)};
  setting[mouseShiftCtrlAltMidClickAction]={"mainwindow/manipulate3d/mouseShiftCtrlAltMidClickAction", static_cast<int>(CTA::None)};
  using CTA=MyTouchWidget::ClickTapAction;
  setting[touchNoneTapAction]=        {"mainwindow/manipulate3d/touchNoneTapAction", static_cast<int>(CTA::SelectTopObject)};
  setting[touchShiftTapAction]=       {"mainwindow/manipulate3d/touchShiftTapAction", static_cast<int>(CTA::SetRotationPointAndCursorSz)};
  setting[touchCtrlTapAction]=        {"mainwindow/manipulate3d/touchCtrlTapAction", static_cast<int>(CTA::ToggleTopObject)};
  setting[touchAltTapAction]=         {"mainwindow/manipulate3d/touchAltTapAction", static_cast<int>(CTA::SelectAnyObject)};
  setting[touchShiftCtrlTapAction]=   {"mainwindow/manipulate3d/touchShiftCtrlTapAction", static_cast<int>(CTA::None)};
  setting[touchShiftAltTapAction]=    {"mainwindow/manipulate3d/touchShiftAltTapAction", static_cast<int>(CTA::None)};
  setting[touchCtrlAltTapAction]=     {"mainwindow/manipulate3d/touchCtrlAltTapAction", static_cast<int>(CTA::ToggleAnyObject)};
  setting[touchShiftCtrlAltTapAction]={"mainwindow/manipulate3d/touchShiftCtrlAltTapAction", static_cast<int>(CTA::None)};
  setting[touchNoneLongTapAction]=        {"mainwindow/manipulate3d/touchNoneLongTapAction", static_cast<int>(CTA::SelectTopObjectAndShowContextMenu)};
  setting[touchShiftLongTapAction]=       {"mainwindow/manipulate3d/touchShiftLongTapAction", static_cast<int>(CTA::None)};
  setting[touchCtrlLongTapAction]=        {"mainwindow/manipulate3d/touchCtrlLongTapAction", static_cast<int>(CTA::ShowContextMenu)};
  setting[touchAltLongTapAction]=         {"mainwindow/manipulate3d/touchAltLongTapAction", static_cast<int>(CTA::SelectAnyObjectAndShowContextMenu)};
  setting[touchShiftCtrlLongTapAction]=   {"mainwindow/manipulate3d/touchShiftCtrlLongTapAction", static_cast<int>(CTA::None)};
  setting[touchShiftAltLongTapAction]=    {"mainwindow/manipulate3d/touchShiftAltLongTapAction", static_cast<int>(CTA::None)};
  setting[touchCtrlAltLongTapAction]=     {"mainwindow/manipulate3d/touchCtrlAltLongTapAction", static_cast<int>(CTA::None)};
  setting[touchShiftCtrlAltLongTapAction]={"mainwindow/manipulate3d/touchShiftCtrlAltLongTapAction", static_cast<int>(CTA::None)};
  using MA=MyTouchWidget::MoveAction;
  setting[touchNoneMove1Action]=        {"mainwindow/manipulate3d/touchNoneMove1Action", static_cast<int>(MA::RotateAboutSySx)};
  setting[touchShiftMove1Action]=       {"mainwindow/manipulate3d/touchShiftMove1Action", static_cast<int>(MA::None)};
  setting[touchCtrlMove1Action]=        {"mainwindow/manipulate3d/touchCtrlMove1Action", static_cast<int>(MA::None)};//mfmf not working pan and pinch is connected!!!!!!!!!!!!!
  setting[touchAltMove1Action]=         {"mainwindow/manipulate3d/touchAltMove1Action", static_cast<int>(MA::ChangeFrame)};
  setting[touchShiftCtrlMove1Action]=   {"mainwindow/manipulate3d/touchShiftCtrlMove1Action", static_cast<int>(MA::CameraAngle)};
  setting[touchShiftAltMove1Action]=    {"mainwindow/manipulate3d/touchShiftAltMove1Action", static_cast<int>(MA::CameraAndRotationPointSz)};
  setting[touchCtrlAltMove1Action]=     {"mainwindow/manipulate3d/touchCtrlAltMove1Action", static_cast<int>(MA::CameraNearPlane)};
  setting[touchShiftCtrlAltMove1Action]={"mainwindow/manipulate3d/touchShiftCtrlAltMove1Action", static_cast<int>(MA::None)};
  setting[touchNoneMove2Action]=        {"mainwindow/manipulate3d/touchNoneMove2Action", static_cast<int>(MA::Translate)};
  setting[touchShiftMove2Action]=       {"mainwindow/manipulate3d/touchShiftMove2Action", static_cast<int>(MA::None)};
  setting[touchCtrlMove2Action]=        {"mainwindow/manipulate3d/touchCtrlMove2Action", static_cast<int>(MA::CursorSz)};//mfmf not working
  setting[touchAltMove2Action]=         {"mainwindow/manipulate3d/touchAltMove2Action", static_cast<int>(MA::None)};
  setting[touchShiftCtrlMove2Action]=   {"mainwindow/manipulate3d/touchShiftCtrlMove2Action", static_cast<int>(MA::None)};
  setting[touchShiftAltMove2Action]=    {"mainwindow/manipulate3d/touchShiftAltMove2Action", static_cast<int>(MA::None)};
  setting[touchCtrlAltMove2Action]=     {"mainwindow/manipulate3d/touchCtrlAltMove2Action", static_cast<int>(MA::None)};
  setting[touchShiftCtrlAltMove2Action]={"mainwindow/manipulate3d/touchShiftCtrlAltMove2Action", static_cast<int>(MA::None)};
  setting[touchNoneMove2ZoomAction]=        {"mainwindow/manipulate3d/touchNoneMove2ZoomAction", static_cast<int>(MA::Zoom)};
  setting[touchShiftMove2ZoomAction]=       {"mainwindow/manipulate3d/touchShiftMove2ZoomAction", static_cast<int>(MA::None)};
  setting[touchCtrlMove2ZoomAction]=        {"mainwindow/manipulate3d/touchCtrlMove2ZoomAction", static_cast<int>(MA::None)};
  setting[touchAltMove2ZoomAction]=         {"mainwindow/manipulate3d/touchAltMove2ZoomAction", static_cast<int>(MA::None)};
  setting[touchShiftCtrlMove2ZoomAction]=   {"mainwindow/manipulate3d/touchShiftCtrlMove2ZoomAction", static_cast<int>(MA::CameraAngle)};
  setting[touchShiftAltMove2ZoomAction]=    {"mainwindow/manipulate3d/touchShiftAltMove2ZoomAction", static_cast<int>(MA::None)};
  setting[touchCtrlAltMove2ZoomAction]=     {"mainwindow/manipulate3d/touchCtrlAltMove2ZoomAction", static_cast<int>(MA::None)};
  setting[touchShiftCtrlAltMove2ZoomAction]={"mainwindow/manipulate3d/touchShiftCtrlAltMove2ZoomAction", static_cast<int>(MA::None)};
  setting[zoomFacPerPixel]={"mainwindow/manipulate3d/zoomFacPerPixel", 1.005};
  setting[zoomFacPerAngle]={"mainwindow/manipulate3d/zoomFacPerAngle", 1.005};
  setting[rotAnglePerPixel]={"mainwindow/manipulate3d/rotAnglePerPixel", 0.2};
  setting[relCursorZPerWheel]={"mainwindow/manipulate3d/relCursorZPerWheel", 0.01};
  setting[relCursorZPerPixel]={"mainwindow/manipulate3d/relCursorZPerPixel", 0.00005};
  setting[pixelPerFrame]={"mainwindow/manipulate3d/pixelPerFrame", 2};
  setting[pickObjectRadius]={"mainwindow/manipulate3d/pickObjectRadius", 3.0};
  setting[inScreenRotateSwitch]={"mainwindow/manipulate3d/inScreenRotateSwitch", 30.0};
  setting[filterType]={"mainwindow/filter/type", 0};
  setting[filterCaseSensitivity]={"mainwindow/filter/casesensitivity", 0};
  setting[transparency]={"mainwindow/sceneGraph/transparency", 2};

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
    StringSetting(QGridLayout *layout, AppSettings::AS key, const QIcon& icon, const QString &name, bool singleLine=true,
                  const QString &tt="", const std::function<void(QString)> &set={});
};
StringSetting::StringSetting(QGridLayout *layout, AppSettings::AS key, const QIcon& icon, const QString &name, bool singleLine,
                             const QString &tt, const std::function<void(QString)> &set) : QWidget(layout->parentWidget()) {
  QFontInfo fontInfo(font());
  int row=layout->rowCount();
  auto *iconLabel=new QLabel;
  iconLabel->setPixmap(icon.pixmap(fontInfo.pixelSize(),fontInfo.pixelSize()));
  layout->addWidget(iconLabel, row, 0);
  auto *label=new QLabel(name, this);
  layout->addWidget(label, row, 1);
  label->setToolTip(tt);
  QLineEdit *lineEdit { nullptr };
  QTextEdit *textEdit { nullptr };
  if(singleLine) {
    lineEdit=new QLineEdit(this);
    lineEdit->setToolTip(tt);
    layout->addWidget(lineEdit, row, 2);
    lineEdit->setText(appSettings->get<QString>(key));
    connect(lineEdit, &QLineEdit::textChanged, [key, set, lineEdit](){
      auto value=lineEdit->text();
      appSettings->set(key, value);
      if(set) set(value);
    });
  }
  else {
    textEdit=new QTextEdit(this);
    textEdit->setToolTip(tt);
    layout->addWidget(textEdit, row, 2);
    textEdit->setText(appSettings->get<QString>(key));
    connect(textEdit, &QTextEdit::textChanged, [key, set, textEdit](){
      auto value=textEdit->toPlainText();
      appSettings->set(key, value);
      if(set) set(value);
    });
  }
}
class ChoiceSetting : public QWidget {
  public:
    ChoiceSetting(QGridLayout *layout, AppSettings::AS key, const QIcon& icon, const QString &name,
                  const vector<pair<QString, QString>> &items, const std::function<void(int)> &set={});
};
ChoiceSetting::ChoiceSetting(QGridLayout *layout, AppSettings::AS key, const QIcon& icon, const QString &name,
                             const vector<pair<QString, QString>> &items, const std::function<void(int)> &set) :
                             QWidget(layout->parentWidget()) {
  QFontInfo fontInfo(font());
  int row=layout->rowCount();
  auto *iconLabel=new QLabel;
  iconLabel->setPixmap(icon.pixmap(fontInfo.pixelSize(),fontInfo.pixelSize()));
  layout->addWidget(iconLabel, row, 0);
  layout->addWidget(new QLabel(name, this), row, 1);
  auto comboBox=new QComboBox(this);
  int idx=0;
  for(auto &i : items) {
    comboBox->insertItem(idx, i.first);
    if(i.second!="")
      comboBox->setItemData(idx, i.second, Qt::ToolTipRole);
    idx++;
  }
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
  auto *showDL=new QPushButton("Color...");
  connect(colorDialog, &QColorDialog::currentColorChanged, [key, set, showDL](const QColor& color){
    appSettings->set(key, color);
    if(set) set(color);
    QPixmap pixmap(100,100);
    pixmap.fill(appSettings->get<QColor>(key));
    QIcon colorIcon(pixmap);
    showDL->setIcon(colorIcon);
  });
  QPixmap pixmap(100,100);
  pixmap.fill(appSettings->get<QColor>(key));
  QIcon colorIcon(pixmap);
  showDL->setIcon(colorIcon);
  connect(showDL, &QPushButton::clicked, this, [colorDialog](){ colorDialog->show(); });
  layout->addWidget(showDL, row, 2);
}

SettingsDialog::SettingsDialog(QWidget *parent) : QDialog(parent) {
  setWindowTitle("Settings");
  setWindowIcon(Utils::QIconCached("settings.svg"));
  auto dialogLayout=new QGridLayout(this);
  setLayout(dialogLayout);
  auto tabWidget=new QTabWidget(this);
  dialogLayout->addWidget(tabWidget);

  auto addTab=[tabWidget](const QIcon &icon, const QString &name) {
    auto tab=new QWidget(tabWidget);
    tabWidget->addTab(tab, icon, name);
    auto tabLayout=new QGridLayout(tabWidget);
    tab->setLayout(tabLayout);
    return tabLayout;
  };

  auto boldLabel=[this](const QString &name, double fac) {
    auto label=new QLabel(name, this);
    auto font=label->font();
    font.setBold(true);
    font.setPointSize(fac*font.pointSize());
    label->setFont(font);
    return label;
  };

  auto addSpace=[](QGridLayout *layout) {
    int row=layout->rowCount();
    layout->addWidget(new QWidget, row, 0, 1,layout->columnCount());
    layout->setRowStretch(row,1);
  };

  auto scene3D=addTab(Utils::QIconCached("viewall.svg"), "3D scene");
  scene3D->setColumnStretch(0, 0);
  scene3D->setColumnStretch(1, 0);
  scene3D->setColumnStretch(2, 1);
  new ColorSetting(scene3D, AppSettings::topBackgroudColor, Utils::QIconCached("bgcolor.svg"), "Top background color:", [](const QColor &color){
    auto rgb=color.rgb();
    MainWindow::getInstance()->bgColor->set1Value(2, qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
    MainWindow::getInstance()->bgColor->set1Value(3, qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
    MainWindow::getInstance()->fgColorTop->set1Value(0, 1-(color.value()+127)/255,1-(color.value()+127)/255,1-(color.value()+127)/255);
    MainWindow::getInstance()->updateScene();
  });
  new ColorSetting(scene3D, AppSettings::bottomBackgroundColor, Utils::QIconCached("bgcolor.svg"), "Bottom background color:", [](const QColor &color){
    auto rgb=color.rgb();
    MainWindow::getInstance()->bgColor->set1Value(0, qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
    MainWindow::getInstance()->bgColor->set1Value(1, qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
    MainWindow::getInstance()->fgColorBottom->set1Value(0, 1-(color.value()+127)/255,1-(color.value()+127)/255,1-(color.value()+127)/255);
    MainWindow::getInstance()->updateScene();
  });
  new DoubleSetting(scene3D, AppSettings::outlineShilouetteEdgeLineWidth, Utils::QIconCached("olselinewidth.svg"), "Outline line width:", "px", [](double value){
    MainWindow::getInstance()->olseDrawStyle->lineWidth.setValue(value);
  }, 0, numeric_limits<double>::max(), 0.1);
  new ColorSetting(scene3D, AppSettings::outlineShilouetteEdgeLineColor, Utils::QIconCached("olsecolor.svg"), "Outline line color:", [](const QColor &value){
    auto rgb=value.rgb();
    MainWindow::getInstance()->olseColor->rgb.set1Value(0, qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
  });
  new DoubleSetting(scene3D, AppSettings::boundingBoxLineWidth, Utils::QIconCached("lines.svg"), "Bounding box line width:", "px", [](double value){
    MainWindow::getInstance()->bboxDrawStyle->lineWidth.setValue(value);
  }, 0, numeric_limits<double>::max(), 0.1);
  new ColorSetting(scene3D, AppSettings::boundingBoxLineColor, Utils::QIconCached("lines.svg"), "Bounding box line color:", [](const QColor &value){
    auto rgb=value.rgb();
    MainWindow::getInstance()->bboxColor->rgb.set1Value(0, qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
  });
  new ChoiceSetting(scene3D, AppSettings::highlightMethod, QIcon(), "Highlight method:", {
    {"Bounding box"                 , "Draw a bbox with highlight-color/-linewidth"},
    {"Transparency"                 , "Increase the transparency of all none highlighted objects"},
    {"Bounding box and transparency", "Both, bbox and transparency"},
  }, [](int value){
    MainWindow::getInstance()->highlightBBox = true;
    MainWindow::getInstance()->highlightTransparency = true;
    MainWindow::getInstance()->highlightObject(nullptr);
    MainWindow::getInstance()->highlightBBox = false;
    MainWindow::getInstance()->highlightTransparency = false;
    if(value == 0 || value == 2)
      MainWindow::getInstance()->highlightBBox = true;
    if(value == 1 || value == 2)
      MainWindow::getInstance()->highlightTransparency = true;
    MainWindow::getInstance()->objectList->itemSelectionChanged();
    if(MainWindow::getInstance()->highlightTransparency)
      MainWindow::getInstance()->highlightItems(MainWindow::getInstance()->objectList->selectedItems());
  });
  new DoubleSetting(scene3D, AppSettings::highlightTransparencyFactor, QIcon(), "Highlight transparency factor:", "", [](double value){
    MainWindow::getInstance()->highlightTransparencyFactor = value;
  }, 0, 1, 0.01);
  new DoubleSetting(scene3D, AppSettings::highlightLineWidth, Utils::QIconCached("lines.svg"), "Highlight line width:", "px", [](double value){
    MainWindow::getInstance()->highlightDrawStyle->lineWidth.setValue(value);
  }, 0, numeric_limits<double>::max(), 0.1);
  new ColorSetting(scene3D, AppSettings::highlightLineColor, Utils::QIconCached("lines.svg"), "Highlight line color:", [](const QColor &value){
    auto rgb=value.rgb();
    MainWindow::getInstance()->highlightColor->rgb.set1Value(0, qRed(rgb)/255.0, qGreen(rgb)/255.0, qBlue(rgb)/255.0);
  });
  new ChoiceSetting(scene3D, AppSettings::complexityType, Utils::QIconCached("complexitytype.svg"), "Complxity type:",
                    {{"Object space", ""}, {"Screen space", ""}, {"Bounding box", ""}}, [](int value){
    MainWindow::getInstance()->complexity->type.setValue( value==0 ? SoComplexity::OBJECT_SPACE :
                                                         (value==1 ? SoComplexity::SCREEN_SPACE :
                                                                     SoComplexity::BOUNDING_BOX));
  });
  new DoubleSetting(scene3D, AppSettings::complexityValue, Utils::QIconCached("complexityvalue.svg"), "Complexity value:", "", [](double value){
    MainWindow::getInstance()->complexity->value.setValue(value);
  }, 0, 1, 0.1);
  new ChoiceSetting(scene3D, AppSettings::transparency, QIcon(), "Transparency type:", {
    {"Blend"                              , ""},
    {"Delayed blend"                      , ""},
    {"Sorted object blend"                , ""},
    {"Sorted object sorted triangle blend", ""},
    {"Sorted layers blend"                , ""}
  }, [](int value){
    MainWindow::getInstance()->glViewer->updateTransperencySetting();
    if(MainWindow::getInstance()->glViewerRight)
      MainWindow::getInstance()->glViewerRight->updateTransperencySetting();
  });
  addSpace(scene3D);

  auto mouseClick=addTab(Utils::QIconCached("mouse.svg"), "Mouse click");
  mouseClick->setRowStretch(0,0);
  auto mouseClickLeftW=new QWidget(this);
  mouseClick->addWidget(boldLabel("Left-Button", 1.3), 0, 0);
  mouseClick->addWidget(mouseClickLeftW, 1, 0);
  auto mouseClickLeft=new QGridLayout;
  mouseClickLeft->setRowStretch(0, 0);
  mouseClickLeft->setColumnStretch(0, 0);
  mouseClickLeft->setColumnStretch(1, 0);
  mouseClickLeft->setColumnStretch(2, 1);
  mouseClickLeftW->setLayout(mouseClickLeft);
  mouseClickLeft->addWidget(boldLabel("Modifier", 1.1), 0, 1);
  mouseClickLeft->addWidget(boldLabel("Action", 1.1), 0, 2);
  auto mouseClickMidW=new QWidget(this);
  mouseClick->addWidget(boldLabel("Mid-Button", 1.3), 0, 1);
  mouseClick->addWidget(mouseClickMidW, 1, 1);
  auto mouseClickMid=new QGridLayout;
  mouseClickMid->setRowStretch(0, 0);
  mouseClickMid->setColumnStretch(0, 0);
  mouseClickMid->setColumnStretch(1, 0);
  mouseClickMid->setColumnStretch(2, 1);
  mouseClickMidW->setLayout(mouseClickMid);
  mouseClickMid->addWidget(boldLabel("Modifier", 1.1), 0, 1);
  mouseClickMid->addWidget(boldLabel("Action", 1.1), 0, 2);
  auto mouseClickRightW=new QWidget(this);
  mouseClick->addWidget(boldLabel("Right-Button", 1.3), 0, 2);
  mouseClick->addWidget(mouseClickRightW, 1, 2);
  auto mouseClickRight=new QGridLayout;
  mouseClickRight->setRowStretch(0, 0);
  mouseClickRight->setColumnStretch(0, 0);
  mouseClickRight->setColumnStretch(1, 0);
  mouseClickRight->setColumnStretch(2, 1);
  mouseClickRightW->setLayout(mouseClickRight);
  mouseClickRight->addWidget(boldLabel("Modifier", 1.1), 0, 1);
  mouseClickRight->addWidget(boldLabel("Action", 1.1), 0, 2);
  #define MOUSECLICK(mod, button) \
    new ChoiceSetting(mouseClick##button, AppSettings::mouse##mod##button##ClickAction, Utils::QIconCached("mouse.svg"), \
        (string(#mod)+":").c_str(), { \
        {"No"                    , "No action."}, \
        {"Select object"         , "Select the object on top of the cursor position."}, \
        {"Toggle object"         , "Toggle the selection of the object on top of the cursor position."}, \
        {"Select any object"     , "Select any object under the cursor position. If more than one object exists a menu is shown."}, \
        {"Toggle any object"     , "Toggle the selection of any object under the cursor position. If more than one object exists a menu is shown."}, \
        {"Show ctx menu"         , "Show the context menu for the currently selected objects."}, \
        {"Sel./show ctx menu"    , "Calls 'Select object' and then 'Show ctx menu'."}, \
        {"Sel. any/show ctx menu", "Calls 'Select any object' and then 'Show ctx menu'"}, \
        {"Set rot. pkt/cursor Sz", "Sets the rotation point of the scene and the cursor screen-z position to the clicked point."}, \
      }, [](int value){ \
      MainWindow::getInstance()->glViewerWG->setMouse##button##ClickAction( \
        MyTouchWidget::Modifier::mod, static_cast<MyTouchWidget::ClickTapAction>(value)); \
      if(MainWindow::getInstance()->dialogStereo) \
        MainWindow::getInstance()->dialogStereo->glViewerWGRight->setMouse##button##ClickAction( \
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
  addSpace(mouseClickLeft);
  addSpace(mouseClickMid);
  addSpace(mouseClickRight);

  auto mouseMove=addTab(Utils::QIconCached("mouse.svg"), "Mouse move");
  mouseMove->setRowStretch(0,0);
  auto mouseMoveLeftW=new QWidget(this);
  mouseMove->addWidget(boldLabel("Left-Button", 1.3), 0, 0);
  mouseMove->addWidget(mouseMoveLeftW, 1, 0);
  auto mouseMoveLeft=new QGridLayout;
  mouseMoveLeft->setRowStretch(0, 0);
  mouseMoveLeft->setColumnStretch(0, 0);
  mouseMoveLeft->setColumnStretch(1, 0);
  mouseMoveLeft->setColumnStretch(2, 1);
  mouseMoveLeftW->setLayout(mouseMoveLeft);
  mouseMoveLeft->addWidget(boldLabel("Modifier", 1.1), 0, 1);
  mouseMoveLeft->addWidget(boldLabel("Action", 1.1), 0, 2);
  auto mouseMoveMidW=new QWidget(this);
  mouseMove->addWidget(boldLabel("Mid-Button", 1.3), 0, 1);
  mouseMove->addWidget(mouseMoveMidW, 1, 1);
  auto mouseMoveMid=new QGridLayout;
  mouseMoveMid->setRowStretch(0, 0);
  mouseMoveMid->setColumnStretch(0, 0);
  mouseMoveMid->setColumnStretch(1, 0);
  mouseMoveMid->setColumnStretch(2, 1);
  mouseMoveMidW->setLayout(mouseMoveMid);
  mouseMoveMid->addWidget(boldLabel("Modifier", 1.1), 0, 1);
  mouseMoveMid->addWidget(boldLabel("Action", 1.1), 0, 2);
  auto mouseMoveRightW=new QWidget(this);
  mouseMove->addWidget(boldLabel("Right-Button", 1.3), 0, 2);
  mouseMove->addWidget(mouseMoveRightW, 1, 2);
  auto mouseMoveRight=new QGridLayout;
  mouseMoveRight->setRowStretch(0, 0);
  mouseMoveRight->setColumnStretch(0, 0);
  mouseMoveRight->setColumnStretch(1, 0);
  mouseMoveRight->setColumnStretch(2, 1);
  mouseMoveRightW->setLayout(mouseMoveRight);
  mouseMoveRight->addWidget(boldLabel("Modifier", 1.1), 0, 1);
  mouseMoveRight->addWidget(boldLabel("Action", 1.1), 0, 2);
  static bool nearPlaneByDistance=getenv("OPENMBV_NEARPLANEBYDISTANCE")!=nullptr;
  pair<QString,QString> nearPlane;
  if(nearPlaneByDistance)
    nearPlane={"Near clip distance", "The distance of the near clipping plane if calculated to <0."};
  else
    nearPlane={"Near clip factor"  , "The normalized (0.1-0.9) near clipping plane factor."};
  #define MOUSEMOVE(mod, button) \
    new ChoiceSetting(mouseMove##button, AppSettings::mouse##mod##button##MoveAction, Utils::QIconCached("mouse.svg"), \
        (string(#mod)+":").c_str(), { \
        {"No"                , "No action."}, \
        {"Change frame"      , "Change the frame number."}, \
        {"Zoom"              , "Zoom in/out."}, \
        {"Camera angle"      , "Change the camera angle (only for perspective camera)."}, \
        {"Cursor S_z"        , "Change the cursor screen z position."}, \
        {"Rotate S_z"        , "Rotate about the screen z axis."}, \
        {"Translate"         , "Translate in screen x and y direction."}, \
        {"Rotate S_y,S_x"    , "Rotate about the screen y and screen x axis."}, \
        {"Rotate W_x,S_x"    , "Rotate about the world x and screen x axis (the world x axis is kept vertical)."}, \
        {"Rotate W_y,S_x"    , "Rotate about the world y and screen x axis (the world y axis is kept vertical)."}, \
        {"Rotate W_z,S_x"    , "Rotate about the world z and screen x axis (the world z axis is kept vertical)."}, \
        {"Cam./rot. pkt S_z" , "Move camera and rotation point in screen z axis."}, \
        {nearPlane.first     , nearPlane.second}, \
      }, [](int value){ \
      MainWindow::getInstance()->glViewerWG->setMouse##button##MoveAction( \
        MyTouchWidget::Modifier::mod, static_cast<MyTouchWidget::MoveAction>(value)); \
      if(MainWindow::getInstance()->dialogStereo) \
        MainWindow::getInstance()->dialogStereo->glViewerWGRight->setMouse##button##MoveAction( \
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
  addSpace(mouseMoveLeft);
  addSpace(mouseMoveMid);
  addSpace(mouseMoveRight);

  auto mouseWheel=addTab(Utils::QIconCached("mouse.svg"), "Mouse wheel");
  mouseWheel->setRowStretch(0,0);
  mouseWheel->setColumnStretch(0, 0);
  mouseWheel->setColumnStretch(1, 0);
  mouseWheel->setColumnStretch(2, 1);
  mouseWheel->addWidget(boldLabel("Modifier", 1.1), 0, 1);
  mouseWheel->addWidget(boldLabel("Action", 1.1), 0, 2);
  #define MOUSEWHEEL(mod) \
    new ChoiceSetting(mouseWheel, AppSettings::mouse##mod##WheelAction, Utils::QIconCached("mouse.svg"), \
        (string(#mod)+":").c_str(), { \
        {"No"          , "No action."}, \
        {"Change frame", "Change the frame number."}, \
        {"Zoom"        , "Zoom in/out."}, \
        {"<N/A>"       , "Not available."}, \
        {"Cursor S_z"  , "Change the cursor screen z position."}, \
        {"Rotate S_z"  , "Rotate about the screen z axis."}, \
        {"<N/A>"       , "Not available."}, \
        {"<N/A>"       , "Not available."}, \
        {"<N/A>"       , "Not available."}, \
        {"<N/A>"       , "Not available."}, \
        {"<N/A>"       , "Not available."}, \
      }, [](int value){ \
      MainWindow::getInstance()->glViewerWG->setMouseWheelAction( \
        MyTouchWidget::Modifier::mod, static_cast<MyTouchWidget::MoveAction>(value)); \
      if(MainWindow::getInstance()->dialogStereo) \
        MainWindow::getInstance()->dialogStereo->glViewerWGRight->setMouseWheelAction( \
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
  addSpace(mouseWheel);

  auto touchTap=addTab(Utils::QIconCached("touch.svg"), "Touch tap");
  touchTap->setRowStretch(0,0);
  auto touchTapTapW=new QWidget(this);
  touchTap->addWidget(boldLabel("Tap", 1.3), 0, 0);
  touchTap->addWidget(touchTapTapW, 1, 0);
  auto touchTapTap=new QGridLayout;
  touchTapTap->setRowStretch(0, 0);
  touchTapTap->setColumnStretch(0, 0);
  touchTapTap->setColumnStretch(1, 0);
  touchTapTap->setColumnStretch(2, 1);
  touchTapTapW->setLayout(touchTapTap);
  touchTapTap->addWidget(boldLabel("Modifier", 1.1), 0, 1);
  touchTapTap->addWidget(boldLabel("Action", 1.1), 0, 2);
  auto touchTapLongTapW=new QWidget(this);
  touchTap->addWidget(boldLabel("Long-Tap", 1.3), 0, 1);
  touchTap->addWidget(touchTapLongTapW, 1, 1);
  auto touchTapLongTap=new QGridLayout;
  touchTapLongTap->setRowStretch(0, 0);
  touchTapLongTap->setColumnStretch(0, 0);
  touchTapLongTap->setColumnStretch(1, 0);
  touchTapLongTap->setColumnStretch(2, 1);
  touchTapLongTapW->setLayout(touchTapLongTap);
  touchTapLongTap->addWidget(boldLabel("Modifier", 1.1), 0, 1);
  touchTapLongTap->addWidget(boldLabel("Action", 1.1), 0, 2);
  #define TOUCHTAP(mod, tap) \
    new ChoiceSetting(touchTap##tap, AppSettings::touch##mod##tap##Action, Utils::QIconCached("touch.svg"), \
        (string(#mod)+":").c_str(), { \
        {"No"                    , "No action."}, \
        {"Select object"         , "Select the object on top of the cursor position."}, \
        {"Toggle object"         , "Toggle the selection of the object on top of the cursor position."}, \
        {"Select any object"     , "Select any object under the cursor position. If more than one object exists a menu is shown."}, \
        {"Toggle any object"     , "Toggle the selection of any object under the cursor position. If more than one object exists a menu is shown."}, \
        {"Show ctx menu"         , "Show the context menu for the currently selected objects."}, \
        {"Sel./show ctx menu"    , "Calls 'Select object' and then 'Show ctx menu'."}, \
        {"Sel. any/show ctx menu", "Calls 'Select any object' and then 'Show ctx menu'"}, \
        {"Set rot. pkt/cursor Sz", "Sets the rotation point of the scene and the cursor screen-z position to the clicked point."}, \
      }, [](int value){ \
      MainWindow::getInstance()->glViewerWG->setTouch##tap##Action( \
        MyTouchWidget::Modifier::mod, static_cast<MyTouchWidget::ClickTapAction>(value)); \
      if(MainWindow::getInstance()->dialogStereo) \
        MainWindow::getInstance()->dialogStereo->glViewerWGRight->setTouch##tap##Action( \
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
  addSpace(touchTapTap);
  addSpace(touchTapLongTap);

  auto touchPan=addTab(Utils::QIconCached("touch.svg"), "Touch pan");
  touchPan->setRowStretch(0,0);
  auto touchPanMove1W=new QWidget(this);
  touchPan->addWidget(boldLabel("1 Finger Pan", 1.3), 0, 0);
  touchPan->addWidget(touchPanMove1W, 1, 0);
  auto touchPanMove1=new QGridLayout;
  touchPanMove1->setRowStretch(0, 0);
  touchPanMove1->setColumnStretch(0, 0);
  touchPanMove1->setColumnStretch(1, 0);
  touchPanMove1->setColumnStretch(2, 1);
  touchPanMove1W->setLayout(touchPanMove1);
  touchPanMove1->addWidget(boldLabel("Modifier", 1.1), 0, 1);
  touchPanMove1->addWidget(boldLabel("Action", 1.1), 0, 2);
  auto touchPanMove2W=new QWidget(this);
  touchPan->addWidget(boldLabel("2 Finger Pan", 1.3), 0, 1);
  touchPan->addWidget(touchPanMove2W, 1, 1);
  auto touchPanMove2=new QGridLayout;
  touchPanMove2->setRowStretch(0, 0);
  touchPanMove2->setColumnStretch(0, 0);
  touchPanMove2->setColumnStretch(1, 0);
  touchPanMove2->setColumnStretch(2, 1);
  touchPanMove2W->setLayout(touchPanMove2);
  touchPanMove2->addWidget(boldLabel("Modifier", 1.1), 0, 1);
  touchPanMove2->addWidget(boldLabel("Action", 1.1), 0, 2);
  #define TOUCHMOVE(mod, tap) \
    new ChoiceSetting(touchPan##tap, AppSettings::touch##mod##tap##Action, Utils::QIconCached("touch.svg"), \
        (string(#mod)+":").c_str(), { \
        {"No"                , "No action."}, \
        {"Change frame"      , "Change the frame number."}, \
        {"<N/A>"             , "Not available."}, \
        {"Camera angle"      , "Change the camera angle (only for perspective camera)"}, \
        {"Cursor S_z"        , "Change the cursor screen z position."}, \
        {"<N/A>"             , "Not available."}, \
        {"Translate"         , "Translate in screen x and y direction."}, \
        {"Rotate S_y,S_x"    , "Rotate about the screen y and screen x axis."}, \
        {"Rotate W_x,S_x"    , "Rotate about the world x and screen x axis (the world x axis is kept vertical)."}, \
        {"Rotate W_y,S_x"    , "Rotate about the world y and screen x axis (the world y axis is kept vertical)."}, \
        {"Rotate W_z,S_x"    , "Rotate about the world z and screen x axis (the world z axis is kept vertical)."}, \
        {"Cam./rot. pkt S_z" , "Move camera and rotation point in screen z axis."}, \
        {nearPlane.first     , nearPlane.second}, \
      }, [](int value){ \
      MainWindow::getInstance()->glViewerWG->setTouch##tap##Action( \
        MyTouchWidget::Modifier::mod, static_cast<MyTouchWidget::MoveAction>(value)); \
      if(MainWindow::getInstance()->dialogStereo) \
        MainWindow::getInstance()->dialogStereo->glViewerWGRight->setTouch##tap##Action( \
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
  addSpace(touchPanMove1);
  addSpace(touchPanMove2);

  auto touchPinch=addTab(Utils::QIconCached("touch.svg"), "Touch pinch");
  touchPinch->setRowStretch(0,0);
  touchPinch->setColumnStretch(0, 0);
  touchPinch->setColumnStretch(1, 0);
  touchPinch->setColumnStretch(2, 1);
  touchPinch->addWidget(boldLabel("Modifier", 1.1), 0, 1);
  touchPinch->addWidget(boldLabel("Action", 1.1), 0, 2);
  #define TOUCHMOVEZOOM(mod) \
    new ChoiceSetting(touchPinch, AppSettings::touch##mod##Move2ZoomAction, Utils::QIconCached("touch.svg"), \
        (string(#mod)+":").c_str(), { \
        {"No"                , "No action."}, \
        {"<N/A>"             , "Not available."}, \
        {"Zoom"              , "Zoom in/out."}, \
        {"Camera angle"      , "Change the camera angle (only for perspective camera)"}, \
        {"<N/A>"             , "Not available."}, \
        {"<N/A>"             , "Not available."}, \
        {"<N/A>"             , "Not available."}, \
        {"<N/A>"             , "Not available."}, \
        {"<N/A>"             , "Not available."}, \
        {"<N/A>"             , "Not available."}, \
        {"<N/A>"             , "Not available."}, \
        {"<N/A>"             , "Not available."}, \
      }, [](int value){ \
      MainWindow::getInstance()->glViewerWG->setTouch##Move2ZoomAction( \
        MyTouchWidget::Modifier::mod, static_cast<MyTouchWidget::MoveAction>(value)); \
      if(MainWindow::getInstance()->dialogStereo) \
        MainWindow::getInstance()->dialogStereo->glViewerWGRight->setTouch##Move2ZoomAction( \
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
  addSpace(touchPinch);

  auto stereoView=addTab(Utils::QIconCached("camerastereo.svg"), "Stereo view");
  stereoView->setColumnStretch(0, 0);
  stereoView->setColumnStretch(1, 0);
  stereoView->setColumnStretch(2, 1);
  new ChoiceSetting(stereoView, AppSettings::stereoType, Utils::QIconCached(""), "Stereo view type:",
                    {{"None", "No stereo view."}, {"Left/Right", "Left/right stereo view for 3D TV screens."}}, [](int value){
    MainWindow::getInstance()->reinit3DView(static_cast<MainWindow::StereoType>(value));
  });
  new DoubleSetting(stereoView, AppSettings::stereoOffset, Utils::QIconCached(""), "Stereo view eye distance:", "m", [](double value){
    MainWindow::getInstance()->setStereoOffset(value);
  }, 0, numeric_limits<double>::max(), 0.01);
  new DoubleSetting(stereoView, AppSettings::stereoAspectRatio, Utils::QIconCached(""), "Stereo view aspect ratio:", "", [](double value){
    MainWindow::getInstance()->glViewer->setAspectRatio(value);
    if(MainWindow::getInstance()->glViewerRight)
      MainWindow::getInstance()->glViewerRight->setAspectRatio(value);
    MainWindow::getInstance()->frame->touch();
  }, 0, numeric_limits<double>::max(), 0.5);
  new ChoiceSetting(stereoView, AppSettings::cameraType, Utils::QIconCached("camera.svg"), "Camera type:",
                    {{"Orthographic", "Orthographic projection, disabled for for stereo view."},
                     {"Perspective" , "Perspective projection, automatically selected for stereo view."}}, [](int value){
    MainWindow::getInstance()->setCameraType(value==0 ? SoOrthographicCamera::getClassTypeId() : SoPerspectiveCamera::getClassTypeId());
    MainWindow::getInstance()->cameraAct->setChecked(value==0);
  });
  addSpace(stereoView);

  auto mouseTouchSettings=addTab(Utils::QIconCached("settings.svg"), "Mouse/Touch settings");
  mouseTouchSettings->setColumnStretch(0, 0);
  mouseTouchSettings->setColumnStretch(1, 0);
  mouseTouchSettings->setColumnStretch(2, 1);
  new ChoiceSetting(mouseTouchSettings, AppSettings::mouseCursor3D, Utils::QIconCached("mouse.svg"), "Mouse cursor type:",
                    {{"2D crosshairs", ""}, {"3D world frame axis", ""}}, [](int value){
    MainWindow::getInstance()->glViewerWG->setCursor3D(value);
  });
  new DoubleSetting(mouseTouchSettings, AppSettings::mouseCursorSize, Utils::QIconCached("mouse.svg"), "Mouse cursor size:", "%", [](double value){
    MainWindow::getInstance()->mouseCursorSizeField->setValue(value);
  }, 0, 100, 1);
  new DoubleSetting(mouseTouchSettings, AppSettings::rotAnglePerPixel, Utils::QIconCached("angle.svg"), "Rotation angle:", "deg/px", [](double value){
    MainWindow::getInstance()->glViewerWG->setRotAnglePerPixel(value);
    if(MainWindow::getInstance()->dialogStereo)
      MainWindow::getInstance()->dialogStereo->glViewerWGRight->setRotAnglePerPixel(value);
  }, numeric_limits<double>::min(), numeric_limits<double>::max(), 0.01);
  new DoubleSetting(mouseTouchSettings, AppSettings::zoomFacPerPixel, Utils::QIconCached("zoom.svg"), "Zoom factor per pixel:", "1/px", [](double value){
    MainWindow::getInstance()->glViewerWG->setZoomFacPerPixel(value);
    if(MainWindow::getInstance()->dialogStereo)
      MainWindow::getInstance()->dialogStereo->glViewerWGRight->setZoomFacPerPixel(value);
  }, 0, numeric_limits<double>::max(), 0.0001);
  new DoubleSetting(mouseTouchSettings, AppSettings::zoomFacPerAngle, Utils::QIconCached("zoom.svg"), "Zoom factor per wheel:", "1/deg", [](double value){
    MainWindow::getInstance()->glViewerWG->setZoomFacPerAngle(value);
    if(MainWindow::getInstance()->dialogStereo)
      MainWindow::getInstance()->dialogStereo->glViewerWGRight->setZoomFacPerAngle(value);
  }, 0, numeric_limits<double>::max(), 0.0001);
  new DoubleSetting(mouseTouchSettings, AppSettings::pickObjectRadius, Utils::QIconCached("target.svg"), "Pick object radius:", "px", [](double value){
    MainWindow::getInstance()->glViewerWG->setPickObjectRadius(value);
    if(MainWindow::getInstance()->dialogStereo)
      MainWindow::getInstance()->dialogStereo->glViewerWGRight->setPickObjectRadius(value);
  }, 0, numeric_limits<double>::max(), 0.1);
  new IntSetting(mouseTouchSettings, AppSettings::tapAndHoldTimeout, Utils::QIconCached("time.svg"), "Tap and hold timeout:", "ms", [](int value){
    MainWindow::getInstance()->glViewerWG->setLongTapInterval(value);
    if(MainWindow::getInstance()->dialogStereo)
      MainWindow::getInstance()->dialogStereo->glViewerWGRight->setLongTapInterval(value);
  });
  new DoubleSetting(mouseTouchSettings, AppSettings::relCursorZPerWheel, Utils::QIconCached("angle.svg"), "Relative cursor-z change per wheel:", "1/wheel", [](double value){
    MainWindow::getInstance()->glViewerWG->setRelCursorZPerWheel(value);
    if(MainWindow::getInstance()->dialogStereo)
      MainWindow::getInstance()->dialogStereo->glViewerWGRight->setRelCursorZPerWheel(value);
  }, 0, 1, 0.001);
  new DoubleSetting(mouseTouchSettings, AppSettings::relCursorZPerPixel, Utils::QIconCached("angle.svg"), "Relative cursor-z change per pixel:", "1/pt", [](double value){
    MainWindow::getInstance()->glViewerWG->setRelCursorZPerPixel(value);
    if(MainWindow::getInstance()->dialogStereo)
      MainWindow::getInstance()->dialogStereo->glViewerWGRight->setRelCursorZPerPixel(value);
  }, 0, 1, 0.0001);
  new IntSetting(mouseTouchSettings, AppSettings::pixelPerFrame, QIcon(), "Pixel per frame:", "pt", [](int value){
    MainWindow::getInstance()->glViewerWG->setPixelPerFrame(value);
    if(MainWindow::getInstance()->dialogStereo)
      MainWindow::getInstance()->dialogStereo->glViewerWGRight->setPixelPerFrame(value);
  });
  new DoubleSetting(mouseTouchSettings, AppSettings::inScreenRotateSwitch, Utils::QIconCached("angle.svg"), "In screen rotate barrier:", "deg", [](double value){
    MainWindow::getInstance()->glViewerWG->setInScreenRotateSwitch(value);
    if(MainWindow::getInstance()->dialogStereo)
      MainWindow::getInstance()->dialogStereo->glViewerWGRight->setInScreenRotateSwitch(value);
  });
  new DoubleSetting(mouseTouchSettings, AppSettings::anglePerKeyPress, Utils::QIconCached("angle.svg"), "Rotation per keypress:", "deg/key", {},
                    numeric_limits<double>::min(), numeric_limits<double>::max(), 0.5);
  addSpace(mouseTouchSettings);

  auto misc=addTab(Utils::QIconCached("settings.svg"), "Misc");
  misc->setColumnStretch(0, 0);
  misc->setColumnStretch(1, 0);
  misc->setColumnStretch(2, 1);
  new IntSetting(misc, AppSettings::hdf5RefreshDelta, Utils::QIconCached("time.svg"), "HDF5 refresh delta:", "ms", [](int value){
    MainWindow::getInstance()->setHDF5RefreshDelta(value);
  });
  new IntSetting(misc, AppSettings::shortAniTime, Utils::QIconCached("time.svg"), "Short animation time:", "ms");
  new DoubleSetting(misc, AppSettings::speedChangeFactor, Utils::QIconCached("speed.svg"), "Animation speed factor:", "1/key", {},
                    0, numeric_limits<double>::max(), 0.01);
  new StringSetting(misc, AppSettings::exportdialog_videocmd, QIcon(), "Video export command:", false,
    "<p>Command to generate the video from a sequence of PNG files:</p>"
    "<ul>"
    "  <li>The absolute path of the input PNG sequence files can be accessed using %I</li>"
    "  <li>The absolute path of the output video file can be accessed using %O</li>"
    "  <li>The bit-rate (in unit Bits per second) can be accessed using %B</li>"
    "  <li>The frame-rate (a floating point number) can be accessed using %F</li>"
    "</ul>"
    "<p>(if using ffmpeg '-c:v libvpx-vp9' (file extension *.webm) is a good codec for web pages and "
    "'-c:v libx264' (file extension *.mp4) is a good codec for MS-Powerpoint))</p>"
    "<p>(note that only the output to stdout of this command is shown in the UI. Pipe stderr to stdout)</p>");
  new StringSetting(misc, AppSettings::exportdialog_videoext, QIcon(), "Video export output filename ext:", true);
  addSpace(misc);
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
