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
#include <QGridLayout>
#include <QLabel>
#include <QLineEdit>
#include <QTreeView>
#include <QListView>
#include <QTableView>
#include <utility>

using namespace std;

namespace OpenMBVGUI {

AbstractViewFilter::AbstractViewFilter(QAbstractItemView *view_, int nameCol_, int typeCol_, const QString &typePrefix_,
                                       std::function<QObject*(const QModelIndex&)> indexToQObject_, int enableRole_) :
  QWidget(view_), view(view_), nameCol(nameCol_), typeCol(typeCol_), typePrefix(typePrefix_), indexToQObject(std::move(indexToQObject_)),
  enableRole(enableRole_) {

  auto *layout=new QGridLayout(this);
  layout->setContentsMargins(0,0,0,0);
  setLayout(layout);
  QLabel *filterL=new QLabel("Filter:");
  if(typeCol==-2) {
    filterL->setToolTip(tr("Filter the view, by applying the given regular expression on the item names (column %1).").arg(nameCol+1));
    filterL->setStatusTip("Filter name column by <regex>.");
  }
  else if(typeCol==-1) {
    filterL->setToolTip(tr("Filter the view, by applying the given regular expression on the item names (column %1).\n"
                           "Or on the type by :<type> or on a derived type by ::<type>.").arg(nameCol+1));
    filterL->setStatusTip("Filter name column by <regex>, or by type :<type>, or by derived type ::<type>");
  }
  else {
    filterL->setToolTip(tr("Filter the view, by applying the given regular expression on the item names (column %1).\n"
                           "Or on the item type (column %2) if the filter starts with ':'.").arg(nameCol+1).arg(typeCol+1));
    filterL->setStatusTip("Filter name column by <regex>, or type column by :<regex>");
  }
  layout->addWidget(filterL, 0, 0);
  filterLE=new QLineEdit;
  filterLE->setToolTip(filterL->toolTip());
  filterLE->setStatusTip(filterL->statusTip());
  layout->addWidget(filterLE, 0, 1);
  connect(filterLE, &QLineEdit::textEdited, this, &AbstractViewFilter::applyFilter);
}

void AbstractViewFilter::setFilter(const QString &filter) {
  filterLE->setText(filter);
  applyFilter();
}

void AbstractViewFilter::applyFilter() {
  // update only if the view is visible (for performance reasons)
  if(!view->isVisible())
    return;
  // do not update if no filter string is set and the filter has not changed
  if(filterLE->text().isEmpty() && oldFilterValue.isEmpty())
    return;
  oldFilterValue=filterLE->text();

  QRegExp filter(filterLE->text());
  // updateMatch will fill the variable match
  updateMatch(view->rootIndex(), filter);
  updateView(view->rootIndex());
  // clear match (no longer requried)
  match.clear();
}

void AbstractViewFilter::updateMatch(const QModelIndex &index, const QRegExp &filter) {
  // do not update the root index
  if(index!=view->rootIndex()) {
    Match &m=match[index];
    // check for matching items
    if(typeCol==-2) {
      // regex search on string on column nameCol
      if(filter.indexIn(view->model()->data(index, Qt::DisplayRole).value<QString>())>=0)
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
        if(filter.indexIn(view->model()->data(index, Qt::DisplayRole).value<QString>())>=0)
          m.me=true;
      }
    }
    else {
      if(filter.pattern().startsWith(":")) { // starting with : => direct type search
        const QModelIndex &colIndex=view->model()->index(index.row(), typeCol, index.parent());
        QRegExp filter2(filter.pattern().mid(1));
        if(filter2.indexIn(view->model()->data(colIndex, Qt::DisplayRole).value<QString>())>=0)
          m.me=true;
      }
      else { // not starting with : => regex search on the string of column nameCol
        if(filter.indexIn(view->model()->data(index, Qt::DisplayRole).value<QString>())>=0)
          m.me=true;
      }
    }
    
    if(m.me) {
      setChildMatchOfParent(index);
      setParentMatchOfChild(index);
    }
  }
  // recursively walk the view
  for(int i=0; i<view->model()->rowCount(index); i++)
    updateMatch(view->model()->index(i, nameCol, index), filter);
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
  // do not update the root index
  if(index!=view->rootIndex()) {
    Match &m=match[index];
    // set hidden (skip further walking of the tree if hidden)
    if(setRowHidden3(qobject_cast<QTreeView*>(view), m, index)) return;
    if(setRowHidden2(qobject_cast<QListView*>(view), m, index)) return;
    if(setRowHidden2(qobject_cast<QTableView*>(view), m, index)) return;
    // set the color of the column nameCol
    bool normalColor=true;
    if(!view->model()->flags(index).testFlag(Qt::ItemIsEnabled) ||
       (view->model()->data(index, enableRole).type()==QVariant::Bool && !view->model()->data(index, enableRole).toBool()))
      normalColor=false;
    QPalette palette;
    if(m.me) {
      if(normalColor)
        view->model()->setData(index, palette.brush(QPalette::Active, QPalette::Text), Qt::ForegroundRole);
      else
        view->model()->setData(index, palette.brush(QPalette::Disabled, QPalette::Text), Qt::ForegroundRole);
    }
    else {
      if(normalColor)
        view->model()->setData(index, QBrush(QColor(255,0,0)), Qt::ForegroundRole);
      else
        view->model()->setData(index, QBrush(QColor(128,0,0)), Qt::ForegroundRole);
    }
    // set expanded
    auto *tree=qobject_cast<QTreeView*>(view);
    if(tree)
      tree->setExpanded(index, m.child);
  }
  // recursively walk the view
  for(int i=0; i<view->model()->rowCount(index); i++)
    updateView(view->model()->index(i, nameCol, index));
}

}
