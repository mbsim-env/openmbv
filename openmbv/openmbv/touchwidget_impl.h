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

#include "touchwidget.h"
#include <cassert>
#include <QMouseEvent>
#include <QTimer>
#include <QApplication>
#include <QTapAndHoldGesture>

namespace OpenMBVGUI {

template<class Widget>
TouchWidget<Widget>::TouchWidget(QWidget *parent, bool handleMouseEvents_, bool handleTouchEvents_) :
                                 Widget(parent), handleMouseEvents(handleMouseEvents_), handleTouchEvents(handleTouchEvents_)  {
  this->setAttribute(Qt::WA_AcceptTouchEvents);
  touchTapDownTimer1=new QTimer(this);
  touchTapDownTimer1->setSingleShot(true);
  touchTapDownTimer1->setInterval(QTapAndHoldGesture::timeout());
  this->connect(touchTapDownTimer1, &QTimer::timeout, this, &TouchWidget::touchLongTapTimedOut1);
}

template<class Widget>
bool TouchWidget<Widget>::event(QEvent *event) {
  // mouse buttions and corresponding functions
  using X=TouchWidget;
  static const std::vector mouseActions {
    //         myButton=idx Qt::MouseButton   click-func           double-click-func          init-func               reset-func               move-func
    std::tuple{ButtonLeft , Qt::LeftButton  , &X::mouseLeftClick , &X::mouseLeftDoubleClick , &X::mouseLeftMoveSave , &X::mouseLeftMoveReset , &X::mouseLeftMove },
    std::tuple{ButtonRight, Qt::RightButton , &X::mouseRightClick, &X::mouseRightDoubleClick, &X::mouseRightMoveSave, &X::mouseRightMoveReset, &X::mouseRightMove},
    std::tuple{ButtonMid  , Qt::MiddleButton, &X::mouseMidClick  , &X::mouseMidDoubleClick  , &X::mouseMidMoveSave  , &X::mouseMidMoveReset  , &X::mouseMidMove  },
  };
  assert(mouseActions.size()==ButtonCOUNT);

  switch(event->type()) {

    case QEvent::MouseButtonPress:
    case QEvent::MouseButtonDblClick: {
      if(!handleMouseEvents)
        return Widget::event(event);
      event->accept();

      auto mouseEvent=static_cast<QMouseEvent*>(event);
      if(mouseEvent->source()!=Qt::MouseEventNotSynthesized) {
        return false;
      }
      for(auto &[myButton_, qtButton_, click_, doubleClick_, init_, reset_, move_]: mouseActions) {
        (void)click_;
        (void)move_;
        if(mouseEvent->button()==qtButton_) {
          // if the last button press is < QApplication::doubleClickInterval() ago -> reset the last click event and emit a double click event
          // (and ignore the following move and release events on this button)
          if(mouseEvent->timestamp()-mouseButtonPressTimestamp[myButton_]<static_cast<unsigned long>(QApplication::doubleClickInterval()) &&
             mouseButtonMaxMoveSqr[myButton_]<QApplication::startDragDistance()*QApplication::startDragDistance()) {
            (this->*reset_)(mouseEvent->modifiers());
            (this->*doubleClick_)(mouseEvent->modifiers(), mouseEvent->pos());
            ignoreMouseMoveRelease[myButton_]=true;
            return true;
          }
          // save position and time of the button press and init the button
          mouseButtonLastPos[myButton_]=mouseButtonPressPos[myButton_]=mouseEvent->pos();
          mouseButtonPressTimestamp[myButton_]=mouseEvent->timestamp();
          mouseButtonMaxMoveSqr[myButton_]=0;
          (this->*init_)(mouseEvent->modifiers());
          return true;
        }
      }
      return true;
    }

    case QEvent::MouseMove: {
      if(!handleMouseEvents)
        return Widget::event(event);
      event->accept();

      auto mouseEvent=static_cast<QMouseEvent*>(event);
      if(mouseEvent->source()!=Qt::MouseEventNotSynthesized) {
        return false;
      }
      for(auto &[myButton_, qtButton_, click_, doubleClick_, init_, reset_, move_]: mouseActions) {
        (void)click_;
        (void)doubleClick_;
        (void)init_;
        (void)reset_;
        if(mouseEvent->buttons() & qtButton_) {
          // if ignore is active for this button -> ignore this move
          if(ignoreMouseMoveRelease[myButton_])
            return true;
          // emit a move event
          auto rel=mouseEvent->pos()-mouseButtonPressPos[myButton_];
          mouseButtonMaxMoveSqr[myButton_]=std::max(mouseButtonMaxMoveSqr[myButton_], QPoint::dotProduct(rel, rel));
          if(mouseButtonLastPos[myButton_]!=mouseEvent->pos()) {
            (this->*move_)(mouseEvent->modifiers(), mouseButtonPressPos[myButton_], mouseEvent->pos());
            mouseButtonLastPos[myButton_]=mouseEvent->pos();
          }
          return true;
        }
      }
      return true;
    }

    case QEvent::MouseButtonRelease: {
      if(!handleMouseEvents)
        return Widget::event(event);
      event->accept();

      auto mouseEvent=static_cast<QMouseEvent*>(event);
      if(mouseEvent->source()!=Qt::MouseEventNotSynthesized) {
        return false;
      }
      for(auto &[myButton_, qtButton_, click_, doubleClick_, init_, reset_, move_]: mouseActions) {
        (void)doubleClick_;
        (void)init_;
        if(mouseEvent->button()==qtButton_) {
          // if ignore is active for this button -> ignore this release and reset the ignore flag
          if(ignoreMouseMoveRelease[myButton_]) {
            ignoreMouseMoveRelease[myButton_]=false;
            return true;
          }
          auto rel=mouseEvent->pos()-mouseButtonPressPos[myButton_];
          mouseButtonMaxMoveSqr[myButton_]=std::max(mouseButtonMaxMoveSqr[myButton_], QPoint::dotProduct(rel, rel));
          // if the last button press is < QApplication::doubleClickInterval() ago -> reset the move event and emit a click event
          if(mouseEvent->timestamp()-mouseButtonPressTimestamp[myButton_]<static_cast<unsigned long>(QApplication::doubleClickInterval()) && 
             mouseButtonMaxMoveSqr[myButton_]<QApplication::startDragDistance()*QApplication::startDragDistance()) {
            // only small move since mouse button press -> reset move and click
            if(mouseButtonLastPos[myButton_]!=mouseButtonPressPos[myButton_]) {
              (this->*reset_)(mouseEvent->modifiers());
              mouseButtonLastPos[myButton_]=mouseButtonPressPos[myButton_];
            }
            (this->*click_)(mouseEvent->modifiers(), mouseEvent->pos());
            return true;
          }
          // emit move event
          if(mouseButtonLastPos[myButton_]!=mouseEvent->pos()) {
            (this->*move_)(mouseEvent->modifiers(), mouseButtonPressPos[myButton_], mouseEvent->pos());
            mouseButtonLastPos[myButton_]=mouseEvent->pos();
          }
          return true;
        }
      }
      return true;
    }

    case QEvent::Wheel: {
      if(!handleMouseEvents)
        return Widget::event(event);
      event->accept();

      auto wheelEvent=static_cast<QWheelEvent*>(event);
      if(wheelEvent->source()!=Qt::MouseEventNotSynthesized) {
        return false;
      }
#if QT_VERSION >= QT_VERSION_CHECK(5, 15, 0)
      mouseWheel(wheelEvent->modifiers(), wheelEvent->angleDelta().y()/8.0, wheelEvent->position().toPoint());
#else
      mouseWheel(wheelEvent->modifiers(), wheelEvent->angleDelta().y()/8.0, wheelEvent->pos());
#endif
      return true;
    }



    case QEvent::TouchBegin: {
      if(!handleTouchEvents)
        return Widget::event(event);
      event->accept();

      auto touchEvent=static_cast<QTouchEvent*>(event);
      if(touchEvent->touchPoints().size()==1) {
        touchTapDownLastPos1=touchTapDownPos1=touchEvent->touchPoints()[0].pos().toPoint();
        touchTapMaxMoveSqr1=0;
        touchCancel1=false;
        touchCancel2=false;
        touchTapTimestamp1=touchEvent->timestamp();
        touchModifiers1=touchEvent->modifiers();
        touchMoveSave1(touchEvent->modifiers());
        touchTapDownTimer1->start();
      }
      if(touchEvent->touchPoints().size()==2) {
        touchCancel1=true;
        touchCancel2=false;
        touchTapDownPos2={touchEvent->touchPoints()[0].pos().toPoint(), touchEvent->touchPoints()[1].pos().toPoint()};
        touchTapDownLastPos1=touchTapDownPos2[0];
        touchTapDownLastPos2=touchTapDownPos2[1];
        touchMoveSave2(touchEvent->modifiers());
      }
      if(touchEvent->touchPoints().size()>=3) {
        touchCancel1=true;
        touchCancel2=true;
      }
      return true;
    }

    case QEvent::TouchUpdate: {
      if(!handleTouchEvents)
        return Widget::event(event);
      event->accept();

      auto touchEvent=static_cast<QTouchEvent*>(event);
      if(touchEvent->touchPoints().size()==1 && !touchCancel1) {
        auto rel=touchEvent->touchPoints()[0].pos().toPoint()-touchTapDownPos1;
        touchTapMaxMoveSqr1=std::max(touchTapMaxMoveSqr1, QPoint::dotProduct(rel, rel));
        if(touchEvent->touchPoints()[0].pos().toPoint()!=touchTapDownLastPos1) {
          touchMove1(touchEvent->modifiers(), touchTapDownPos1, touchEvent->touchPoints()[0].pos().toPoint());
          touchTapDownLastPos1=touchEvent->touchPoints()[0].pos().toPoint();
        }
      }
      if(touchEvent->touchPoints().size()==2 && !touchCancel2) {
        if(!touchCancel1) { // 1 to 2 points
          touchCancel1=true;
          touchMoveReset1(touchEvent->modifiers());
          touchTapDownPos2={touchEvent->touchPoints()[0].pos().toPoint(), touchEvent->touchPoints()[1].pos().toPoint()};
          touchMoveSave2(touchEvent->modifiers());
        }
        if(touchEvent->touchPoints()[0].pos().toPoint()!=touchTapDownLastPos1 ||
           touchEvent->touchPoints()[1].pos().toPoint()!=touchTapDownLastPos2) {
          touchMove2(touchEvent->modifiers(), touchTapDownPos2, {touchEvent->touchPoints()[0].pos().toPoint(),  touchEvent->touchPoints()[1].pos().toPoint()});
          touchTapDownLastPos1=touchEvent->touchPoints()[0].pos().toPoint();
          touchTapDownLastPos2=touchEvent->touchPoints()[1].pos().toPoint();
        }
      }
      if(touchEvent->touchPoints().size()>=3) {
        if(!touchCancel1) { // 1 to 3 points
          touchCancel1=true;
          touchMoveReset1(touchEvent->modifiers());
        }
        if(!touchCancel2) { // 2 to 3 points
          touchCancel2=true;
          touchMoveReset2(touchEvent->modifiers());
        }
      }
      return true;
    }

    case QEvent::TouchEnd: {
      if(!handleTouchEvents)
        return Widget::event(event);
      event->accept();

      auto touchEvent=static_cast<QTouchEvent*>(event);
      touchTapDownTimer1->stop();
      if(touchEvent->touchPoints().size()==1 && !touchCancel1) {
        auto pos=touchEvent->touchPoints()[0].pos().toPoint();
        auto rel=pos-touchTapDownPos1;
        touchTapMaxMoveSqr1=std::max(touchTapMaxMoveSqr1, QPoint::dotProduct(rel, rel));
        auto relLastClick=touchTapLastClickPos-pos;
        if(!touchCancel1 && 
           touchEvent->timestamp()-touchTapTimestamp1<static_cast<unsigned long>(QApplication::doubleClickInterval()) && 
           touchTapMaxMoveSqr1<QApplication::startDragDistance()*QApplication::startDragDistance() &&
           touchEvent->timestamp()-touchTapLastClick<static_cast<unsigned long>(QApplication::doubleClickInterval()) &&
           QPoint::dotProduct(relLastClick, relLastClick)<QApplication::startDragDistance()*QApplication::startDragDistance()) {
          touchMoveReset1(touchEvent->modifiers());
          touchDoubleTap(touchEvent->modifiers(), touchEvent->touchPoints()[0].pos().toPoint());
        }
        else if(!touchCancel1 && 
           touchEvent->timestamp()-touchTapTimestamp1<static_cast<unsigned long>(QApplication::doubleClickInterval()) && 
           touchTapMaxMoveSqr1<QApplication::startDragDistance()*QApplication::startDragDistance()) {
          touchMoveReset1(touchEvent->modifiers());
          touchTapLastClick=touchEvent->timestamp();
          touchTapLastClickPos=pos;
          touchTap(touchEvent->modifiers(), touchEvent->touchPoints()[0].pos().toPoint());
        }
      }
      return true;
    }

    case QEvent::TouchCancel: {
      if(!handleTouchEvents)
        return Widget::event(event);
      event->accept();

      auto touchEvent=static_cast<QTouchEvent*>(event);
      touchTapDownTimer1->stop();
      if(!touchCancel1)
        touchMoveReset1(touchEvent->modifiers());
      if(!touchCancel2)
        touchMoveReset2(touchEvent->modifiers());
      return true;
    }



    default:
      return Widget::event(event);
  };
}

template<class Widget>
void TouchWidget<Widget>::touchLongTapTimedOut1() {
  if(touchTapMaxMoveSqr1>=QApplication::startDragDistance()*QApplication::startDragDistance() || touchCancel1)
    return;
  touchMoveReset1(touchModifiers1);
  touchLongTap(touchModifiers1, touchTapDownPos1);
  touchCancel1=true;
}

}
