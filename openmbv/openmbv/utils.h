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

#ifndef _OPENMBVGUI_UTILS_H_
#define _OPENMBVGUI_UTILS_H_

#include <QIcon>
#include <QDialog>
#include <Inventor/C/errors/debugerror.h> // workaround a include order bug in Coin-3.1.3
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoScale.h>
#include <Inventor/SbRotation.h>
#include <Inventor/nodes/SoTriangleStripSet.h>
#include <Inventor/fields/SoMFColor.h>
#include <string>
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoIndexedFaceSet.h>
#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN // GL/glu.h includes windows.h on Windows -> avoid full header -> WIN32_LEAN_AND_MEAN
#  endif
#endif
#include <GL/glu.h>
#include <openmbvcppinterface/objectfactory.h>
#include <openmbvcppinterface/object.h>
#include <QTreeWidgetItem>
#include <QSettings>
#include <unordered_map>
#include <boost/date_time/posix_time/posix_time.hpp>

#ifdef _WIN32
#  define CALLMETHOD __stdcall
#else
#  define CALLMETHOD
#endif

namespace OpenMBVGUI {

class MainWindow;

template<class T>
class SoSharedPtr {
  public:
    SoSharedPtr() = default;
    SoSharedPtr(const SoSharedPtr& src) {
      ptr = src.ptr;
      ptr->ref();
    }
    SoSharedPtr(SoSharedPtr&& src) noexcept {
      ptr = src.ptr;
      src.ptr=nullptr;
    }
    SoSharedPtr& operator=(const SoSharedPtr& src) {
      if(ptr) ptr->unref();
      ptr = src.ptr;
      ptr->ref();
      return *this;
    }
    SoSharedPtr& operator=(SoSharedPtr&& src) noexcept {
      if(ptr) ptr->unref();
      ptr = src.ptr;
      src.ptr=nullptr;
      return *this;
    }
    SoSharedPtr(T* p) : ptr(p) {
      if(ptr) ptr->ref();
    }
    ~SoSharedPtr() {
      if(ptr) ptr->unref();
    }
    void reset(T* p) {
      if(ptr) ptr->unref();
      ptr = p;
      if(ptr) ptr->ref();
    }
    operator bool() const {
      return ptr!=nullptr;
    }
    T* get() const {
      return ptr;
    }
  private:
    T* ptr { nullptr };
};

/** Utilitiy class */
class Utils : virtual public fmatvec::Atom {
  friend MainWindow;
  public:
    // INITIALIZATION

    /** initialize the Utils class. Must be called before any member is used. */
    static void initialize();

    static void deinitialize();



    // HELPER FUNCTIONS

    /** Use QIconCached(basefilename) instead of QIcon(filename) everywhere
     * to cache the parsing of e.g. SVG files. This lead to a speedup
     * (at app init) by a factor of 11 in my test case.
     * (note that only a base-filename is passed as parameter NOT a relative or absolute filename) */
    static const QIcon& QIconCached(const std::string &basefilename);

    /** Use SoDBreadAllFileNameCached(filename) instead of SoDB::readAll(filename) everywhere
     * to cache the iv-file parsing and scene generation.
     * hash is, beside filename, part of the key for the cache. */
    static SoSeparator* SoDBreadAllFileNameCached(const std::string &filename, size_t hash=0, const std::function<void(SoInput&)> &inFunc=nullptr);

    /** Use SoDBreadAllContentCached(filename) instead of SoDB::readAll(filename) everywhere
     * to cache the iv-content parsing and scene generation.
     * hash is, beside content, part of the key for the cache. */
    static SoSeparator* SoDBreadAllContentCached(const std::string &content, size_t hash=0, const std::function<void(SoInput&)> &inFunc=nullptr);

    /** Get the node named name being a child or grandchild of sep */
    static SoNode* getChildNodeByName(SoGroup *sep, const SbName &name);

    static SoMFColor soFrameDefaultColor;
    /** Convenienc function to draw a frame */
    static SoSeparator* soFrame(double size, double offset, bool pickBBoxAble, SoScale *&scale,
                                const SbColor &xCol=SbColor(1,0,0), const SbColor &yCol=SbColor(0,1,0), const SbColor &zCol=SbColor(0,0,1));
    /** Convenienc function to draw a frame */
    static SoSeparator* soFrame(double size, double offset, bool pickBBoxAble,
                                const SbColor &xCol=SbColor(1,0,0), const SbColor &yCol=SbColor(0,1,0), const SbColor &zCol=SbColor(0,0,1)) {
      SoScale *scale; return soFrame(size, offset, pickBBoxAble, scale, xCol,yCol,zCol);
    }

    /** Convenienc function to convert cardan angles to a rotation matrix */
    static SbRotation cardan2Rotation(const SbVec3f& c);
    /** Convenienc function to convert a rotation matrix to cardan angles */
    static SbVec3f rotation2Cardan(const SbRotation& R);

    template<class T>
    static void visitTreeWidgetItems(QTreeWidgetItem *root, std::function<void (T)> func, bool onlySelected=false);

    static std::string getIconPath();
    static std::string getXMLDocPath();
    static std::string getDocPath();

    static void enableTouch(QWidget *w);


    // TESSELATION
    static GLUtesselator *tess();


    using FactoryElement = std::tuple<QIcon, std::string, std::function<std::shared_ptr<OpenMBV::Object> ()>>;
    static std::shared_ptr<OpenMBV::Object> createObjectEditor(const std::vector<FactoryElement> &factory,
                                                                 const std::vector<std::string> &existingNames,
                                                                 const std::string &title);
    template<class T>
    static std::function<std::shared_ptr<OpenMBV::Object>()> factory() {
      return static_cast<std::shared_ptr<T>(*)()>(&OpenMBV::ObjectFactory::create<T>);
    }


  private:
    struct SoDeleteSeparator {
      SoSharedPtr<SoSeparator> sep;
      boost::posix_time::ptime fileTime;
    };
    static std::unordered_map<size_t, SoDeleteSeparator> ivCache;
    static std::unordered_map<std::string, QIcon> iconCache;

    // INITIALIZATION
    static bool initialized;

    // TESSELATION
    static GLenum tessType;
    static int tessNumVertices;
    static SoTriangleStripSet *tessTriangleStrip;
    static SoIndexedFaceSet *tessTriangleFan;
    static SoCoordinate3 *tessCoord;
    static void CALLMETHOD tessBeginCB(GLenum type, void *data);
    static void CALLMETHOD tessVertexCB(GLdouble *vertex);
    static void CALLMETHOD tessEndCB();
};

template<class T>
void Utils::visitTreeWidgetItems(QTreeWidgetItem *root, std::function<void (T)> func, bool onlySelected) {
  for(int i=0; i<root->childCount(); i++)
    visitTreeWidgetItems(root->child(i), func, onlySelected);
  if((!onlySelected || root->isSelected()) && dynamic_cast<T>(root))
    func(static_cast<T>(root));
}

class IgnoreWheelEventFilter : public QObject {
  public:
    static IgnoreWheelEventFilter instance;
  protected:
    bool eventFilter(QObject *watched, QEvent *event) override;
};

class AppSettings {
  public:
    enum AS {
      hdf5RefreshDelta,
      cameraType,
      stereoType,
      stereoOffset,
      stereoAspectRatio,
      tapAndHoldTimeout,
      outlineShilouetteEdgeLineWidth,
      outlineShilouetteEdgeLineColor,
      boundingBoxLineWidth,
      boundingBoxLineColor,
      highlightLineWidth,
      highlightLineColor,
      complexityType,
      complexityValue,
      topBackgroudColor,
      bottomBackgroundColor,
      anglePerKeyPress,
      speedChangeFactor,
      shortAniTime,
      mainwindow_geometry,
      mainwindow_state,
      settingsDialog_geometry,
      exportdialog_resolutionfactor,
      exportdialog_usescenecolor,
      exportdialog_fps,
      exportdialog_filename_png,
      exportdialog_filename_video,
      exportdialog_bitrate,
      exportdialog_videocmd,
      exportdialog_videoext,
      propertydialog_geometry,
      dialogstereo_geometry,
      mouseCursor3D,
      mouseCursorSize,
      mouseNoneLeftMoveAction,
      mouseShiftLeftMoveAction,
      mouseCtrlLeftMoveAction,
      mouseAltLeftMoveAction,
      mouseShiftCtrlLeftMoveAction,
      mouseShiftAltLeftMoveAction,
      mouseCtrlAltLeftMoveAction,
      mouseShiftCtrlAltLeftMoveAction,
      mouseNoneRightMoveAction,
      mouseShiftRightMoveAction,
      mouseCtrlRightMoveAction,
      mouseAltRightMoveAction,
      mouseShiftCtrlRightMoveAction,
      mouseShiftAltRightMoveAction,
      mouseCtrlAltRightMoveAction,
      mouseShiftCtrlAltRightMoveAction,
      mouseNoneMidMoveAction,
      mouseShiftMidMoveAction,
      mouseCtrlMidMoveAction,
      mouseAltMidMoveAction,
      mouseShiftCtrlMidMoveAction,
      mouseShiftAltMidMoveAction,
      mouseCtrlAltMidMoveAction,
      mouseShiftCtrlAltMidMoveAction,
      mouseNoneWheelAction,
      mouseShiftWheelAction,
      mouseCtrlWheelAction,
      mouseAltWheelAction,
      mouseShiftCtrlWheelAction,
      mouseShiftAltWheelAction,
      mouseCtrlAltWheelAction,
      mouseShiftCtrlAltWheelAction,
      mouseNoneLeftClickAction,
      mouseShiftLeftClickAction,
      mouseCtrlLeftClickAction,
      mouseAltLeftClickAction,
      mouseShiftCtrlLeftClickAction,
      mouseShiftAltLeftClickAction,
      mouseCtrlAltLeftClickAction,
      mouseShiftCtrlAltLeftClickAction,
      mouseNoneRightClickAction,
      mouseShiftRightClickAction,
      mouseCtrlRightClickAction,
      mouseAltRightClickAction,
      mouseShiftCtrlRightClickAction,
      mouseShiftAltRightClickAction,
      mouseCtrlAltRightClickAction,
      mouseShiftCtrlAltRightClickAction,
      mouseNoneMidClickAction,
      mouseShiftMidClickAction,
      mouseCtrlMidClickAction,
      mouseAltMidClickAction,
      mouseShiftCtrlMidClickAction,
      mouseShiftAltMidClickAction,
      mouseCtrlAltMidClickAction,
      mouseShiftCtrlAltMidClickAction,
      touchNoneTapAction,
      touchShiftTapAction,
      touchCtrlTapAction,
      touchAltTapAction,
      touchShiftCtrlTapAction,
      touchShiftAltTapAction,
      touchCtrlAltTapAction,
      touchShiftCtrlAltTapAction,
      touchNoneLongTapAction,
      touchShiftLongTapAction,
      touchCtrlLongTapAction,
      touchAltLongTapAction,
      touchShiftCtrlLongTapAction,
      touchShiftAltLongTapAction,
      touchCtrlAltLongTapAction,
      touchShiftCtrlAltLongTapAction,
      touchNoneMove1Action,
      touchShiftMove1Action,
      touchCtrlMove1Action,
      touchAltMove1Action,
      touchShiftCtrlMove1Action,
      touchShiftAltMove1Action,
      touchCtrlAltMove1Action,
      touchShiftCtrlAltMove1Action,
      touchNoneMove2Action,
      touchShiftMove2Action,
      touchCtrlMove2Action,
      touchAltMove2Action,
      touchShiftCtrlMove2Action,
      touchShiftAltMove2Action,
      touchCtrlAltMove2Action,
      touchShiftCtrlAltMove2Action,
      touchNoneMove2ZoomAction,
      touchShiftMove2ZoomAction,
      touchCtrlMove2ZoomAction,
      touchAltMove2ZoomAction,
      touchShiftCtrlMove2ZoomAction,
      touchShiftAltMove2ZoomAction,
      touchCtrlAltMove2ZoomAction,
      touchShiftCtrlAltMove2ZoomAction,
      zoomFacPerPixel,
      zoomFacPerAngle,
      rotAnglePerPixel,
      relCursorZPerWheel,
      relCursorZPerPixel,
      pixelPerFrame,
      pickObjectRadius,
      inScreenRotateSwitch,
      filterType,
      filterCaseSensitivity,
      transparency,
      SIZE,
    };
    AppSettings();
    ~AppSettings();
    template<class T> T get(AS as) {
      auto &value=setting[as].second;
      return value.value<T>();
    }
    template<class T> void set(AS as, const T& newValue, bool callQSettings=false) {
      auto &[str, value]=setting[as];
      value=newValue;
      if(callQSettings)
        qSettings.setValue(str, value);
    }
    static constexpr auto format{QSettings::IniFormat};
    static constexpr auto scope{QSettings::UserScope};
    static constexpr auto organization{"mbsim-env"};
    static constexpr auto application{"openmbv"};
    static constexpr auto organizationDomain{"www.mbsim-env.de"};
  private:
    QSettings qSettings;
    std::vector<std::pair<QString, QVariant>> setting;
};

extern std::unique_ptr<AppSettings> appSettings;

class SettingsDialog : public QDialog {
  public:
    SettingsDialog(QWidget *parent);
    void closeEvent(QCloseEvent *event) override;
    void showEvent(QShowEvent *event) override;
};

}

#endif
