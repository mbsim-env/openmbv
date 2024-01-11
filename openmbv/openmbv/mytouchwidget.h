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

#ifndef _OPENMBVGUI_MYTOUCHWIDGET_H_
#define _OPENMBVGUI_MYTOUCHWIDGET_H_

#include "Inventor/SbVec3f.h"
#include "Inventor/SbRotation.h"
#include "Inventor/lists/SoPickedPointList.h"
#include "touchwidget.h"
#include <QWidget>
#include <variant>

namespace OpenMBVGUI {

class Body;

class MyTouchWidget : public TouchWidget<QWidget> {
  public:
    MyTouchWidget(QWidget *parent);
    bool event(QEvent *event) override;

    enum class Modifier { None, Shift, Ctrl, Alt, ShiftCtrl, ShiftAlt, CtrlAlt, ShiftCtrlAlt, SIZE };
    template<class T>
    class ModArray {
      public:
        T operator[](Modifier mod) const { return a[static_cast<size_t>(mod)]; }
        T& operator[](Modifier mod) { return a[static_cast<size_t>(mod)]; }
      private:
        std::array<T, static_cast<size_t>(Modifier::SIZE)> a;
    };
    enum class ClickTapAction {
      None,
      SelectTopObject,
      ToggleTopObject,
      SelectAnyObject,
      ToggleAnyObject,
      ShowContextMenu,
      SelectTopObjectAndShowContextMenu,
      SelectAnyObjectAndShowContextMenu,
      SetFocalPoint,
    };
    enum class MoveAction {
      None,
      ChangeFrame, // 1D
      Zoom, // 1D
      CameraDistFromFocalPoint, // 1D
      CurserSz, // 1D
      RotateAboutSz, // 1D
      Translate, // 2D
      RotateAboutSySx, // 2D
      RotateAboutWxSx, // 2D
      RotateAboutWySx, // 2D
      RotateAboutWzSx, // 2D
      CameraAndFocalPointSz, // 1D
      CameraNearPlane, // 1D
    };

    int getVerticalAxis() { return verticalAxis; }
    void setMouseLeftClickAction (Modifier mod, ClickTapAction act) { mouseLeftClickAction [mod]=act; }
    void setMouseRightClickAction(Modifier mod, ClickTapAction act) { mouseRightClickAction[mod]=act; }
    void setMouseMidClickAction  (Modifier mod, ClickTapAction act) { mouseMidClickAction  [mod]=act; }
    void setMouseLeftMoveAction  (Modifier mod, MoveAction act) { mouseLeftMoveAction [mod]=act; setVerticalAxis(act); }
    void setMouseRightMoveAction (Modifier mod, MoveAction act) { mouseRightMoveAction[mod]=act; setVerticalAxis(act); }
    void setMouseMidMoveAction   (Modifier mod, MoveAction act) { mouseMidMoveAction  [mod]=act; setVerticalAxis(act); }
    void setMouseWheelAction     (Modifier mod, MoveAction act) { mouseWheelAction  [mod]=act; setVerticalAxis(act); }
    void setTouchTapAction       (Modifier mod, ClickTapAction act) { touchTapAction    [mod]=act; }
    void setTouchLongTapAction   (Modifier mod, ClickTapAction act) { touchLongTapAction[mod]=act; }
    void setTouchMove1Action     (Modifier mod, MoveAction act) { touchMove1Action[mod]=act; setVerticalAxis(act); }
    void setTouchMove2Action     (Modifier mod, MoveAction act) { touchMove2Action[mod]=act; setVerticalAxis(act); }
    void setTouchMove2ZoomAction (Modifier mod, MoveAction act) { touchMove2ZoomAction[mod]=act; setVerticalAxis(act); }
    void setZoomFacPerPixel(double value) { zoomFacPerPixel=value; }
    void setZoomFacPerAngle(double value) { zoomFacPerAngle=value; }
    void setRotAnglePerPixel(double value) { rotAnglePerPixel=value; }
    void setPickObjectRadius(double value) { pickObjectRadius=value; }
    void setInScreenRotateSwitch(double value) { inScreenRotateSwitch=value; }
    void setRelCursorZPerWheel(double value) { relCursorZPerWheel=value; }
    void setRelCursorZPerPixel(double value) { relCursorZPerPixel=value; }
    void setPixelPerFrame(int value) { pixelPerFrame=value; }
  protected:
    // functions for mouse events
    void mouseLeftClick(Qt::KeyboardModifiers modifiers, const QPoint &pos) override;
    void mouseRightClick(Qt::KeyboardModifiers modifiers, const QPoint &pos) override;
    void mouseMidClick(Qt::KeyboardModifiers modifiers, const QPoint &pos) override;
    void mouseLeftDoubleClick(Qt::KeyboardModifiers modifiers, const QPoint &pos) override;
    void mouseRightDoubleClick(Qt::KeyboardModifiers modifiers, const QPoint &pos) override;
    void mouseMidDoubleClick(Qt::KeyboardModifiers modifiers, const QPoint &pos) override;
    void mouseLeftMoveSave(Qt::KeyboardModifiers modifiers, const QPoint& initialPos) override;
    void mouseRightMoveSave(Qt::KeyboardModifiers modifiers, const QPoint& initialPos) override;
    void mouseMidMoveSave(Qt::KeyboardModifiers modifiers, const QPoint& initialPos) override;
    void mouseLeftMoveReset(Qt::KeyboardModifiers modifiers) override;
    void mouseRightMoveReset(Qt::KeyboardModifiers modifiers) override;
    void mouseMidMoveReset(Qt::KeyboardModifiers modifiers) override;
    void mouseLeftMove(Qt::KeyboardModifiers modifiers, const QPoint &initialPos, const QPoint &pos) override;
    void mouseRightMove(Qt::KeyboardModifiers modifiers, const QPoint &initialPos, const QPoint &pos) override;
    void mouseMidMove(Qt::KeyboardModifiers modifiers, const QPoint &initialPos, const QPoint &pos) override;
    void mouseWheel(Qt::KeyboardModifiers modifiers, double relAngle, const QPoint &pos) override;
    // functions for touch events
    void touchTap(Qt::KeyboardModifiers modifiers, const QPoint &pos) override;
    void touchDoubleTap(Qt::KeyboardModifiers modifiers, const QPoint &pos) override;
    void touchLongTap(Qt::KeyboardModifiers modifiers, const QPoint &pos) override;
    void touchMoveSave1(Qt::KeyboardModifiers modifiers, const QPoint &initialPos) override;
    void touchMoveSave2(Qt::KeyboardModifiers modifiers, const std::array<QPoint, 2> &initialPos) override;
    void touchMoveReset1(Qt::KeyboardModifiers modifiers) override;
    void touchMoveReset2(Qt::KeyboardModifiers modifiers) override;
    void touchMove1(Qt::KeyboardModifiers modifiers, const QPoint &initialPos, const QPoint &pos) override;
    void touchMove2(Qt::KeyboardModifiers modifiers, const std::array<QPoint, 2> &initialPos, const std::array<QPoint, 2> &pos) override;
  private:
    ModArray<ClickTapAction> mouseLeftClickAction;
    ModArray<ClickTapAction> mouseRightClickAction;
    ModArray<ClickTapAction> mouseMidClickAction;
    ModArray<MoveAction> mouseLeftMoveAction;
    ModArray<MoveAction> mouseRightMoveAction;
    ModArray<MoveAction> mouseMidMoveAction;
    ModArray<MoveAction> mouseWheelAction;
    ModArray<ClickTapAction> touchTapAction;
    ModArray<ClickTapAction> touchLongTapAction;
    ModArray<MoveAction> touchMove1Action;
    ModArray<MoveAction> touchMove2Action;
    ModArray<MoveAction> touchMove2ZoomAction;
    float zoomFacPerPixel;
    float zoomFacPerAngle;
    float rotAnglePerPixel;
    float pickObjectRadius;
    float inScreenRotateSwitch;
    float relCursorZ = 0.5; // only used by the left eye view, the right eye view uses relCursorZ from the left eye view
    float relCursorZPerWheel;
    float relCursorZPerPixel;
    int pixelPerFrame;

    int verticalAxis { 2 };
    void setVerticalAxis(MoveAction act);

    double touchMove2RotateInScreenPlane;
    SbVec3f initialTranslateCameraPos;
    float initialZoomCameraHeight;
    float initialZoomCameraHeightAngle;
    SbVec3f initialZoomCameraPos;
    float initialZoomCameraNearPlane;
    float initialZoomCameraFocalDistance;
    SbRotation initialRotateCameraOri;
    SbVec3f initialRotateCameraPos;
    SbVec3f initialRotateCameraToPos;
    int initialFrame;

    std::vector<std::pair<Body*, std::vector<SbVec3f>>> getObjectsByRay(const QPoint &pos);
    int createObjectListMenu(const std::vector<Body*>& pickedObject);

    void selectObject(const QPoint &pos, bool toggle, bool showMenuForAll);
    void selectObjectAndShowContextMenu(const QPoint &pos, bool showMenuForAll);
    void setFocalPoint(const QPoint &pos, Body *body=nullptr);
    void openPropertyDialog(const QPoint &pos);
    void rotateInit(const QPoint &initialPos);
    void rotateReset();
    void rotateAboutSySx(const QPoint &rel, const QPoint &pos);
    void rotateAboutWSx (const QPoint &rel, const QPoint &pos, int axisIdx); // used by three belos functions
    void rotateAboutWxSx(const QPoint &rel, const QPoint &pos);
    void rotateAboutWySx(const QPoint &rel, const QPoint &pos);
    void rotateAboutWzSx(const QPoint &rel, const QPoint &pos);
    void rotateAboutSz(double relAngle, bool relIoInitial=true);
    void translateInit();
    void translateReset();
    void translate(const QPoint &rel);
    void zoomInit();
    void zoomReset();
    void zoomCameraAngle(float fac);
    void cameraDistFromFocalPoint(int change);
    void cameraAndFocalPointSz(const QPoint &rel, const QPoint &pos);
    void cameraNearPlane(const QPoint &rel, const QPoint &pos);
    void cursorSz(float change, const QPoint &pos);
    void changeFrame(int steps, bool rel=true);
    void updateCursorPos(const QPoint &mousePos);
};

}

#endif
