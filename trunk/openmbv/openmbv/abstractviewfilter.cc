/*
    OpenMBV - Open Multi Body Viewer.
    Copyright (C) 2009 Markus Friedrich

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program; if not, write to the Free Software
    Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include "config.h"
#include <abstractviewfilter.h>
#include <QtGui/QGridLayout>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QTreeView>
#include <QtGui/QListView>
#include <QtGui/QTableView>

using namespace std;

namespace OpenMBVGUI {

AbstractViewFilter::AbstractViewFilter(QAbstractItemView *view_, int nameCol_, int typeCol_, const QString &typePrefix_, boost::function<QObject*(const QModelIndex&)> indexToQObject_) : QWidget(view_), view(view_), nameCol(nameCol_), typeCol(typeCol_), typePrefix(typePrefix_), indexToQObject(indexToQObject_) {
  QGridLayout *layout=new QGridLayout(this);
  layout->setContentsMargins(0,0,0,0);
  setLayout(layout);
  QLabel *filterL=new QLabel("Filter:");
  if(typeCol==-2) {
    filterL->setToolTip(tr("Filter the view, by applying the given regular expression on the item names (col. %1).").arg(nameCol));
    filterL->setStatusTip("Filter name by <regex>");
  }
  else if(typeCol==-1) {
    filterL->setToolTip(tr("Filter the view, by applying the given regular expression on the item names (col. %1),\n"
                           "or\n"
                           "when the filter starts with : by the given type of the items or,\n"
                           "when the filter starts with :: by the given type or derived type of the items.").arg(nameCol));
    filterL->setStatusTip("Filter name by <regex>, or type by :<type>, or derived type by ::<type>");
  }
  else {
    filterL->setToolTip(tr("Filter the view, by applying the given regular expression on the item names (col. %1),\n"
                           "or\n"
                           "when the filter starts with : by the given type (col. %2) of the items.").arg(nameCol, typeCol));
    filterL->setStatusTip("Filter name by <regex>, or type by :<type>");
  }
  layout->addWidget(filterL, 0, 0);
  filterLE=new QLineEdit;
  filterLE->setToolTip(filterL->toolTip());
  filterLE->setStatusTip(filterL->statusTip());
  layout->addWidget(filterLE, 0, 1);
  connect(filterLE, SIGNAL(returnPressed()), this, SLOT(applyFilter()));
}

void AbstractViewFilter::setFilter(const QString &filter) {
  filterLE->setText(filter);
  applyFilter();
}

void AbstractViewFilter::applyFilter() {
  // update only if the view is visible
  if(!view->isVisible())
    return;

  QRegExp filter(filterLE->text());
  // updateMatch will fill the variable match
  updateMatch(view->rootIndex(), filter);
  updateView(view->rootIndex());
  // clear match (no longer requried)
  match.clear();
}

void AbstractViewFilter::updateMatch(const QModelIndex &index, const QRegExp &filter) {
  // recursively walk the view
  for(int i=0; i<view->model()->rowCount(index); i++)
    updateMatch(view->model()->index(i, nameCol, index), filter);
  // do nothing but walking the view for the invisibleRootItem element
  if(index==view->rootIndex())
    return;

  Match &m=match[index];
  // check for matching items
  if(typeCol==-2) {
    // regex search on string on column nameCol
    if(filter.indexIn(view->model()->data(index, Qt::EditRole).value<QString>())>=0)
      m.me=true;
  }
  else if(typeCol==-1) {
    if(filter.pattern().startsWith("::")) { // starting with :: => inherited type search
      QObject *obj=indexToQObject(index);
      if(obj && obj->inherits((typePrefix+filter.pattern().mid(2)).toStdString().c_str()))
        m.me=true;
    }
    else if(filter.pattern().startsWith(":")) { // starting with : => direct type search
      QObject *obj=indexToQObject(index);
      if(obj) {
        QString str=obj->metaObject()->className();
        str=str.replace(typePrefix, "");
        if(str==filter.pattern().mid(1))
          m.me=true;
      }
    }
    else { // not starting with : or :: => regex search on the string of column nameCol
      if(filter.indexIn(view->model()->data(index, Qt::EditRole).value<QString>())>=0)
        m.me=true;
    }
  }
  else {
    if(filter.pattern().startsWith(":")) { // starting with : => direct type search
      const QModelIndex &colIndex=view->model()->index(index.row(), typeCol, index.parent());
      if(typePrefix+filter.pattern().mid(1)==view->model()->data(colIndex, Qt::EditRole).value<QString>())
        m.me=true;
    }
    else { // not starting with : or :: => regex search on the string of column nameCol
      if(filter.indexIn(view->model()->data(index, Qt::EditRole).value<QString>())>=0)
        m.me=true;
    }
  }
  
  if(m.me) {
    setChildMatchOfParent(index);
    setParentMatchOfChild(index);
  }
}

void AbstractViewFilter::setChildMatchOfParent(const QModelIndex &index) {
  const QModelIndex &p=index.parent();
  if(!p.isValid())
    return;
  Match &m=match[p];
  m.child=true;
  setChildMatchOfParent(p);
}

void AbstractViewFilter::setParentMatchOfChild(const QModelIndex &index) {
  for(int i=0; i<view->model()->rowCount(index); i++) {
    const QModelIndex &c=view->model()->index(i, nameCol, index);
    Match &m=match[c];
    m.parent=true;
    setParentMatchOfChild(c);
  }
}

void AbstractViewFilter::updateView(const QModelIndex &index) {
  // recursively walk the view
  for(int i=0; i<view->model()->rowCount(index); i++)
    updateView(view->model()->index(i, nameCol, index));
  // do nothing but walking the view for the invisibleRootItem element
  if(index==view->rootIndex())
    return;

  Match &m=match[index];
  // set hidden
  if(setRowHidden3(qobject_cast<QTreeView*>(view), m, index)) return;
  if(setRowHidden2(qobject_cast<QListView*>(view), m, index)) return;
  if(setRowHidden2(qobject_cast<QTableView*>(view), m, index)) return;
  // set the color of the column nameCol
  QPalette palette;
  if(m.me) {
    if((view->model()->flags(index) & Qt::ItemIsEnabled) > 0)
      view->model()->setData(index, palette.brush(QPalette::Active, QPalette::Text), Qt::ForegroundRole);
    else
      view->model()->setData(index, palette.brush(QPalette::Disabled, QPalette::Text), Qt::ForegroundRole);
  }
  else {
    if((view->model()->flags(index) & Qt::ItemIsEnabled) > 0)
      view->model()->setData(index, QBrush(QColor(255,0,0)), Qt::ForegroundRole);
    else
      view->model()->setData(index, QBrush(QColor(128,0,0)), Qt::ForegroundRole);
  }
  // set expanded
  QTreeView *tree=qobject_cast<QTreeView*>(view);
  if(tree) {
    if(m.child)
      tree->setExpanded(index, true);
    else
      tree->setExpanded(index, false);
  }
}

}
