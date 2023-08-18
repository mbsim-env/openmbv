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
SoSeparator* Utils::soFrame(double size, double offset, bool pickBBoxAble, SoScale *&scale) {
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
  col->rgb=SbColor(1, 0, 0);
  sep->addChild(col);
  line=new SoLineSet;
  line->startIndex.setValue(0);
  line->numVertices.setValue(2);
  sep->addChild(line);

  // y-axis
  col=new SoBaseColor;
  col->rgb=SbColor(0, 1, 0);
  sep->addChild(col);
  line=new SoLineSet;
  line->startIndex.setValue(2);
  line->numVertices.setValue(2);
  sep->addChild(line);

  // z-axis
  col=new SoBaseColor;
  col->rgb=SbColor(0, 0, 1);
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
  return SbRotation(SbMatrix(
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
  ));
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
    if(dialog.exec()!=QDialog::Accepted) return std::shared_ptr<OpenMBV::Object>();
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
  setting[stereoType]={"mainwindow/sceneGraph/stereoType", 0};
  setting[stereoOffset]={"mainwindow/sceneGraph/stereoOffset", 0.1};
  setting[stereoAnaglyphColorMask]={"mainwindow/sceneGraph/stereoAnaglyphColorMask", "100011"};
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
  using MMA=MyTouchWidget::MouseMoveAction;
  setting[mouseLeftMoveAction]={"mainwindow/manipulate3d/mouseLeftMoveAction", static_cast<int>(MMA::Rotate)};
  setting[mouseRightMoveAction]={"mainwindow/manipulate3d/mouseRightMoveAction", static_cast<int>(MMA::Translate)};
  setting[mouseMidMoveAction]={"mainwindow/manipulate3d/mouseMidMoveAction", static_cast<int>(MMA::Zoom)};
  using MCA=MyTouchWidget::MouseClickAction;
  setting[mouseLeftClickAction]={"mainwindow/manipulate3d/mouseLeftClickAction", static_cast<int>(MCA::Select)};
  setting[mouseRightClickAction]={"mainwindow/manipulate3d/mouseRightClickAction", static_cast<int>(MCA::Context)};
  setting[mouseMidClickAction]={"mainwindow/manipulate3d/mouseMidClickAction", static_cast<int>(MCA::SeekToPoint)};
  using TTA=MyTouchWidget::TouchTapAction;
  setting[touchTapAction]={"mainwindow/manipulate3d/touchTapAction", static_cast<int>(TTA::Select)};
  setting[touchLongTapAction]={"mainwindow/manipulate3d/touchLongTapAction", static_cast<int>(TTA::Context)};
  using TMA=MyTouchWidget::TouchMoveAction;
  setting[touchMove1Action]={"mainwindow/manipulate3d/touchMove1Action", static_cast<int>(TMA::Rotate)};
  setting[touchMove2Action]={"mainwindow/manipulate3d/touchMove2Action", static_cast<int>(TMA::Translate)};
  setting[zoomFacPerPixel]={"mainwindow/manipulate3d/zoomFacPerPixel", 1.005};
  setting[rotAnglePerPixel]={"mainwindow/manipulate3d/rotAnglePerPixel", 0.2};
  setting[pickObjectRadius]={"mainwindow/manipulate3d/pickObjectRadius", 3.0};
  setting[inScreenRotateSwitch]={"mainwindow/manipulate3d/inScreenRotateSwitch", 30.0};
  setting[filterType]={"mainwindow/filter/type", 0};
  setting[filterCaseSensitivity]={"mainwindow/filter/casesensitivity", 0};

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
  new ChoiceSetting(layout, AppSettings::stereoType, Utils::QIconCached(""), "Stereo view type:",
                    {"None", "AnaGlyph", "Quad buffer", "Interleaved rows", "Interleaved columns"}, [](int value){
    SoQtViewer::StereoType t;
    switch(value) {
      case 0: t=SoQtMyViewer::STEREO_NONE; break;
      case 1: t=SoQtMyViewer::STEREO_ANAGLYPH; break;
      case 2: t=SoQtMyViewer::STEREO_QUADBUFFER; break;
      case 3: t=SoQtMyViewer::STEREO_INTERLEAVED_ROWS; break;
      case 4: t=SoQtMyViewer::STEREO_INTERLEAVED_COLUMNS; break;
      default: t=SoQtMyViewer::STEREO_NONE; break;
    }
    if(value!=0)
      MainWindow::getInstance()->glViewer->setCameraType(SoPerspectiveCamera::getClassTypeId());
    MainWindow::getInstance()->glViewer->setStereoType(t);
  });
  new DoubleSetting(layout, AppSettings::stereoOffset, Utils::QIconCached(""), "Stereo view eye distance:", "m", [](double value){
    MainWindow::getInstance()->glViewer->setStereoOffset(value);
  }, 0, numeric_limits<double>::max(), 0.01);
  new StringSetting(layout, AppSettings::stereoAnaglyphColorMask, Utils::QIconCached(""), "Stereo view color mask:", [](QString value){
    SbBool left[]={value[0]!='0', value[1]!='0', value[2]!='0'};
    SbBool right[]={value[3]!='0', value[4]!='0', value[5]!='0'};
    MainWindow::getInstance()->glViewer->setAnaglyphStereoColorMasks(left, right);
  });
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
  new ChoiceSetting(layout, AppSettings::mouseLeftClickAction, Utils::QIconCached("mouse.svg"), "Mouse left click:",
                    {"Select object", "Show context menu of object", "Seek scene to point under cursor"}, [](int value){
    MainWindow::getInstance()->glViewerWG->setMouseLeftClickAction(static_cast<MyTouchWidget::MouseClickAction>(value));
  });
  new ChoiceSetting(layout, AppSettings::mouseRightClickAction, Utils::QIconCached("mouse.svg"), "Mouse right click:",
                    {"Select object", "Show context menu of object", "Seek scene to point under cursor"}, [](int value){
    MainWindow::getInstance()->glViewerWG->setMouseRightClickAction(static_cast<MyTouchWidget::MouseClickAction>(value));
  });
  new ChoiceSetting(layout, AppSettings::mouseMidClickAction, Utils::QIconCached("mouse.svg"), "Mouse middle click:",
                    {"Select object", "Show context menu of object", "Seek scene to point under cursor"}, [](int value){
    MainWindow::getInstance()->glViewerWG->setMouseMidClickAction(static_cast<MyTouchWidget::MouseClickAction>(value));
  });
  new ChoiceSetting(layout, AppSettings::mouseLeftMoveAction, Utils::QIconCached("mouse.svg"), "Mouse left move:",
                    {"Rotate scene in screen axis", "Translate scene", "Zoom scene"}, [](int value){
    MainWindow::getInstance()->glViewerWG->setMouseLeftMoveAction(static_cast<MyTouchWidget::MouseMoveAction>(value));
  });
  new ChoiceSetting(layout, AppSettings::mouseRightMoveAction, Utils::QIconCached("mouse.svg"), "Mouse right move:",
                    {"Rotate scene in screen axis", "Translate scene", "Zoom scene"}, [](int value){
    MainWindow::getInstance()->glViewerWG->setMouseRightMoveAction(static_cast<MyTouchWidget::MouseMoveAction>(value));
  });
  new ChoiceSetting(layout, AppSettings::mouseMidMoveAction, Utils::QIconCached("mouse.svg"), "Mouse middle move:",
                    {"Rotate scene in screen axis", "Translate scene", "Zoom scene"}, [](int value){
    MainWindow::getInstance()->glViewerWG->setMouseMidMoveAction(static_cast<MyTouchWidget::MouseMoveAction>(value));
  });
  new ChoiceSetting(layout, AppSettings::touchTapAction, Utils::QIconCached("touch.svg"), "Touch tap:",
                    {"Select object", "Show context menu of object"}, [](int value){
    MainWindow::getInstance()->glViewerWG->setTouchTapAction(static_cast<MyTouchWidget::TouchTapAction>(value));
  });
  new ChoiceSetting(layout, AppSettings::touchLongTapAction, Utils::QIconCached("touch.svg"), "Touch long tap",
                    {"Select object", "Show context menu of object"}, [](int value){
    MainWindow::getInstance()->glViewerWG->setTouchLongTapAction(static_cast<MyTouchWidget::TouchTapAction>(value));
  });
  new ChoiceSetting(layout, AppSettings::touchMove1Action, Utils::QIconCached("touch.svg"), "Touch 1-finger pan:",
                    {"Rotate scene in screen axis", "Translate scene"}, [](int value){
    MainWindow::getInstance()->glViewerWG->setTouchMove1Action(static_cast<MyTouchWidget::TouchMoveAction>(value));
  });
  new ChoiceSetting(layout, AppSettings::touchMove2Action, Utils::QIconCached("touch.svg"), "Touch 2-finger pan:",
                    {"Rotate scene in screen axis", "Translate scene"}, [](int value){
    MainWindow::getInstance()->glViewerWG->setTouchMove2Action(static_cast<MyTouchWidget::TouchMoveAction>(value));
  });
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
