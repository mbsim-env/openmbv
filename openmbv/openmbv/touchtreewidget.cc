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

#include "touchtreewidget.h"
#include <QGestureEvent>
#include <QApplication>
#include <QTimer>

namespace OpenMBVGUI {

TouchTreeWidget::TouchTreeWidget(QWidget *parent) : TouchWidget<QTreeWidget>(parent, false) {
}

void TouchTreeWidget::touchTap(Qt::KeyboardModifiers modifiers, const QPoint &pos) {
  setCurrentItem(itemAt(pos));
}

void TouchTreeWidget::touchDoubleTap(Qt::KeyboardModifiers modifiers, const QPoint &pos) {
  setCurrentItem(itemAt(pos));
  itemDoubleClicked(itemAt(pos), currentColumn());
}

void TouchTreeWidget::touchLongTap(Qt::KeyboardModifiers modifiers, const QPoint &pos) {
  setCurrentItem(itemAt(pos));
  customContextMenuRequested(pos);
}


}
