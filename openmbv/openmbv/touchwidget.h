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

#ifndef _OPENMBVGUI_TOUCHWIDGET_H_
#define _OPENMBVGUI_TOUCHWIDGET_H_

#include <array>
#include <QPoint>

class QMouseEvent;
class QTouchEvent;
class QWidget;
class QEvent;
class QTimer;

namespace OpenMBVGUI {

//! Base class for SoQtMyViewer just for hanlding Qt-events (transmforming it to camera movement)
template<class Widget>
class TouchWidget : public Widget {
  public:
    TouchWidget(QWidget *parent, bool handleMouseEvents_=true, bool handleTouchEvents_=true);
    bool event(QEvent *event) override;
  private:
    bool handleMouseEvents;
    bool handleTouchEvents;
  private:
    // variables for mouse events
    enum MouseButton { ButtonLeft, ButtonRight, ButtonMid, ButtonCOUNT };
    std::array<QPoint       , ButtonCOUNT> mouseButtonPressPos;
    std::array<QPoint       , ButtonCOUNT> mouseButtonLastPos;
    std::array<unsigned long, ButtonCOUNT> mouseButtonPressTimestamp { 0, 0, 0 };
    std::array<int          , ButtonCOUNT> mouseButtonMaxMoveSqr;
    std::array<bool         , ButtonCOUNT> ignoreMouseMoveRelease { false, false, false };
    bool mouseLeftMoveActive { false }; // DEBUG only
    bool mouseRightMoveActive { false }; // DEBUG only
    bool mouseMidMoveActive { false }; // DEBUG only
  protected:
    // functions for mouse events
    virtual void mouseLeftClick(Qt::KeyboardModifiers modifiers, const QPoint &pos) {}
    virtual void mouseRightClick(Qt::KeyboardModifiers modifiers, const QPoint &pos) {}
    virtual void mouseMidClick(Qt::KeyboardModifiers modifiers, const QPoint &pos) {}
    virtual void mouseLeftDoubleClick(Qt::KeyboardModifiers modifiers, const QPoint &pos) {}
    virtual void mouseRightDoubleClick(Qt::KeyboardModifiers modifiers, const QPoint &pos) {}
    virtual void mouseMidDoubleClick(Qt::KeyboardModifiers modifiers, const QPoint &pos) {}
    virtual void mouseLeftMoveSave(Qt::KeyboardModifiers modifiers, const QPoint &initialPos) {}
    virtual void mouseRightMoveSave(Qt::KeyboardModifiers modifiers, const QPoint &initialPos) {}
    virtual void mouseMidMoveSave(Qt::KeyboardModifiers modifiers, const QPoint &initialPos) {}
    virtual void mouseLeftMoveReset(Qt::KeyboardModifiers modifiers) {}
    virtual void mouseRightMoveReset(Qt::KeyboardModifiers modifiers) {}
    virtual void mouseMidMoveReset(Qt::KeyboardModifiers modifiers) {}
    virtual void mouseLeftMove(Qt::KeyboardModifiers modifiers, const QPoint &initialPos, const QPoint &pos) {}
    virtual void mouseRightMove(Qt::KeyboardModifiers modifiers, const QPoint &initialPos, const QPoint &pos) {}
    virtual void mouseMidMove(Qt::KeyboardModifiers modifiers, const QPoint &initialPos, const QPoint &pos) {}
    virtual void mouseWheel(Qt::KeyboardModifiers modifiers, double relAngle, const QPoint &pos) {}

  private:
    // variables for touch events
    QTimer *touchTapDownTimer1;
    QPoint touchTapDownPos1;
    unsigned long touchTapTimestamp1 { 0 };
    unsigned long touchTapLastClick { 0 };
    QPoint touchTapLastClickPos;
    int touchTapMaxMoveSqr1;
    bool touchCancel1;
    bool touchCancel2;
    Qt::KeyboardModifiers touchModifiers1;
    void touchLongTapTimedOut1();
    std::array<QPoint, 2> touchTapDownPos2;
    QPoint touchTapDownLastPos1;
    QPoint touchTapDownLastPos2;
    bool touchMoveActive1 { false }; // DEBUG only
    bool touchMoveActive2 { false }; // DEBUG only
  protected:
    // functions for touch events
    virtual void touchTap(Qt::KeyboardModifiers modifiers, const QPoint &pos) {}
    virtual void touchDoubleTap(Qt::KeyboardModifiers modifiers, const QPoint &pos) {}
    virtual void touchLongTap(Qt::KeyboardModifiers modifiers, const QPoint &pos) {}
    virtual void touchMoveSave1(Qt::KeyboardModifiers modifiers, const QPoint &initialPos) {}
    virtual void touchMoveSave2(Qt::KeyboardModifiers modifiers, const std::array<QPoint, 2> &initialPos) {}
    virtual void touchMoveReset1(Qt::KeyboardModifiers modifiers) {}
    virtual void touchMoveReset2(Qt::KeyboardModifiers modifiers) {}
    virtual void touchMove1(Qt::KeyboardModifiers modifiers, const QPoint &initialPos, const QPoint &pos) {}
    virtual void touchMove2(Qt::KeyboardModifiers modifiers, const std::array<QPoint, 2> &initialPos, const std::array<QPoint, 2> &pos) {}
};

}

#endif
