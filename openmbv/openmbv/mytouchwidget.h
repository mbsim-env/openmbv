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

namespace OpenMBVGUI {

class Body;

class MyTouchWidget : public TouchWidget<QWidget> {
  public:
    MyTouchWidget(QWidget *parent);

    enum class MouseClickAction { Select, Context, SeekToPoint };
    enum class MouseMoveAction { Rotate, Translate, Zoom };
    enum class TouchTapAction { Select, Context };
    enum class TouchMoveAction { Rotate, Translate };

    void setMouseLeftMoveAction  (MouseMoveAction  act) { mouseLeftMoveAction  =act; }
    void setMouseRightMoveAction (MouseMoveAction  act) { mouseRightMoveAction =act; }
    void setMouseMidMoveAction   (MouseMoveAction  act) { mouseMidMoveAction   =act; }
    void setMouseLeftClickAction (MouseClickAction act) { mouseLeftClickAction =act; }
    void setMouseRightClickAction(MouseClickAction act) { mouseRightClickAction=act; }
    void setMouseMidClickAction  (MouseClickAction act) { mouseMidClickAction  =act; }
    void setTouchTapAction       (TouchTapAction   act) { touchTapAction       =act; }
    void setTouchLongTapAction   (TouchTapAction   act) { touchLongTapAction   =act; }
    void setTouchMove1Action     (TouchMoveAction  act) { touchMove1Action     =act; }
    void setTouchMove2Action     (TouchMoveAction  act) { touchMove2Action     =act; }
    void setZoomFacPerPixel(double value) { zoomFacPerPixel=value; }
    void setRotAnglePerPixel(double value) { rotAnglePerPixel=value; }
    void setPickObjectRadius(double value) { pickObjectRadius=value; }
    void setInScreenRotateSwitch(double value) { inScreenRotateSwitch=value; }
  protected:
    // functions for mouse events
    void mouseLeftClick(Qt::KeyboardModifiers modifiers, const QPoint &pos) override;
    void mouseRightClick(Qt::KeyboardModifiers modifiers, const QPoint &pos) override;
    void mouseMidClick(Qt::KeyboardModifiers modifiers, const QPoint &pos) override;
    void mouseLeftDoubleClick(Qt::KeyboardModifiers modifiers, const QPoint &pos) override;
    void mouseRightDoubleClick(Qt::KeyboardModifiers modifiers, const QPoint &pos) override;
    void mouseMidDoubleClick(Qt::KeyboardModifiers modifiers, const QPoint &pos) override;
    void mouseLeftMoveSave(Qt::KeyboardModifiers modifiers) override;
    void mouseRightMoveSave(Qt::KeyboardModifiers modifiers) override;
    void mouseMidMoveSave(Qt::KeyboardModifiers modifiers) override;
    void mouseLeftMoveReset(Qt::KeyboardModifiers modifiers) override;
    void mouseRightMoveReset(Qt::KeyboardModifiers modifiers) override;
    void mouseMidMoveReset(Qt::KeyboardModifiers modifiers) override;
    void mouseLeftMove(Qt::KeyboardModifiers modifiers, const QPoint &initialPos, const QPoint &pos) override;
    void mouseRightMove(Qt::KeyboardModifiers modifiers, const QPoint &initialPos, const QPoint &pos) override;
    void mouseMidMove(Qt::KeyboardModifiers modifiers, const QPoint &initialPos, const QPoint &pos) override;
    void mouseWheel(Qt::KeyboardModifiers modifiers, double relAngle) override;
    // functions for touch events
    void touchTap(Qt::KeyboardModifiers modifiers, const QPoint &pos) override;
    void touchDoubleTap(Qt::KeyboardModifiers modifiers, const QPoint &pos) override;
    void touchLongTap(Qt::KeyboardModifiers modifiers, const QPoint &pos) override;
    void touchMoveSave1(Qt::KeyboardModifiers modifiers) override;
    void touchMoveSave2(Qt::KeyboardModifiers modifiers) override;
    void touchMoveReset1(Qt::KeyboardModifiers modifiers) override;
    void touchMoveReset2(Qt::KeyboardModifiers modifiers) override;
    void touchMove1(Qt::KeyboardModifiers modifiers, const QPoint &initialPos, const QPoint &pos) override;
    void touchMove2(Qt::KeyboardModifiers modifiers, const std::array<QPoint, 2> &initialPos, const std::array<QPoint, 2> &pos) override;
  private:
    MouseClickAction mouseLeftClickAction;
    MouseClickAction mouseRightClickAction;
    MouseClickAction mouseMidClickAction;
    MouseMoveAction mouseLeftMoveAction;
    MouseMoveAction mouseRightMoveAction;
    MouseMoveAction mouseMidMoveAction;
    TouchTapAction touchTapAction;
    TouchTapAction touchLongTapAction;
    TouchMoveAction touchMove1Action;
    TouchMoveAction touchMove2Action;
    float zoomFacPerPixel;
    float rotAnglePerPixel;
    float pickObjectRadius;
    float inScreenRotateSwitch;

    double touchMove2RotateInScreenPlane;
    SbVec3f initialTranslateCameraPos;
    float initialZoomCameraHeight;
    float initialZoomCameraHeightAngle;
    SbVec3f initialZoomCameraPos;
    float initialZoomCameraFocalDistance;
    SbRotation initialRotateCameraOri;
    SbVec3f initialRotateCameraPos;
    SbVec3f initialRotateCameraToPos;

    std::vector<std::pair<Body*, std::vector<SbVec3f>>> getObjectsByRay(const QPoint &pos);
    int createObjectListMenu(const std::vector<Body*>& pickedObject);

    void selectObject(const QPoint &pos, bool toggle, bool showMenuForAll);
    void selectObjectAndShowContextMenu(const QPoint &pos, bool showMenuForAll);
    void seekToPoint(const QPoint &pos, Body *body=nullptr);
    void openPropertyDialog(const QPoint &pos);
    void rotateInit();
    void rotateReset();
    void rotateInScreenAxis(const QPoint &rel);
    void rotateInScreenPlane(double relAngle);
    void translateInit();
    void translateReset();
    void translate(const QPoint &rel);
    void zoomInit();
    void zoomReset();
    void zoomCameraAngle(int change);
    void zoomCameraFocalDist(int change);
    void changeFrame(int steps);
};

}

#endif
