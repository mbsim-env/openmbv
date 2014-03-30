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
#include <filteredtreewidget.h>
#include <QtGui/QGridLayout>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>

using namespace std;

namespace OpenMBVGUI {

FilteredTreeWidget::FilteredTreeWidget(QWidget *parent, int nameCol_, int typeCol_, const QString &typePrefix_) : QTreeWidget(parent), nameCol(nameCol_), typeCol(typeCol_), typePrefix(typePrefix_) {
  QGridLayout *layout=new QGridLayout(this);
  layout->setContentsMargins(0,0,0,0);
  widget=new QWidget(this);
  widget->setLayout(layout);
  QLabel *filterL=new QLabel("Filter:");
  if(typeCol==-2) {
    filterL->setToolTip(tr("Filter the tree view, by applying the given regular expression on the item names (col. %1).").arg(nameCol));
    filterL->setStatusTip("Filter name by <regex>");
  }
  else if(typeCol==-1) {
    filterL->setToolTip(tr("Filter the tree view, by applying the given regular expression on the item names (col. %1),\n"
                           "or\n"
                           "when the filter starts with : by the given type of the items or,\n"
                           "when the filter starts with :: by the given type or derived type of the items.").arg(nameCol));
    filterL->setStatusTip("Filter name by <regex>, or type by :<type>, or derived type by ::<type>");
  }
  else {
    filterL->setToolTip(tr("Filter the tree view, by applying the given regular expression on the item names (col. %1),\n"
                           "or\n"
                           "when the filter starts with : by the given type (col. %2) of the items.").arg(nameCol, typeCol));
    filterL->setStatusTip("Filter name by <regex>, or type by :<type>");
  }
  layout->addWidget(filterL, 0, 0);
  filterLE=new QLineEdit(this);
  filterLE->setToolTip(filterL->toolTip());
  filterLE->setStatusTip(filterL->statusTip());
  layout->addWidget(filterLE, 0, 1);
  connect(filterLE, SIGNAL(returnPressed()), this, SLOT(applyFilter()));
}

void FilteredTreeWidget::setFilter(const QString &filter) {
  filterLE->setText(filter);
  applyFilter();
}

void FilteredTreeWidget::applyFilter() {
  QRegExp filter(filterLE->text());
  match.clear();
  updateMatch(invisibleRootItem(), filter);
  updateView(invisibleRootItem());
}

void FilteredTreeWidget::updateMatch(QTreeWidgetItem *item, const QRegExp &filter) {
  // recursively walk the tree
  for(int i=0; i<item->childCount(); i++)
    updateMatch(item->child(i), filter);
  // do nothing but walking the tree for the invisibleRootItem element
  if(item==invisibleRootItem())
    return;

  Match &m=match[item];
  // check for matching items
  if(typeCol==-2) {
    // regex search on string on column nameCol
    if(filter.indexIn(item->text(nameCol))>=0)
      m.me=true;
  }
  else if(typeCol==-1) {
    if(filter.pattern().startsWith("::")) { // starting with :: => inherited type search
      QObject *obj=dynamic_cast<QObject*>(item);
      if(obj && obj->inherits((typePrefix+filter.pattern().mid(2)).toStdString().c_str()))
        m.me=true;
    }
    else if(filter.pattern().startsWith(":")) { // starting with : => direct type search
      QObject *obj=dynamic_cast<QObject*>(item);
      if(obj) {
        QString str=obj->metaObject()->className();
        str=str.replace(typePrefix, "");
        if(str==filter.pattern().mid(1))
          m.me=true;
      }
    }
    else { // not starting with : or :: => regex search on the string of column nameCol
      if(filter.indexIn(item->text(nameCol))>=0)
        m.me=true;
    }
  }
  else {
    if(filter.pattern().startsWith(":")) { // starting with : => direct type search
      if(typePrefix+filter.pattern().mid(1)==item->text(typeCol))
        m.me=true;
    }
    else { // not starting with : or :: => regex search on the string of column nameCol
      if(filter.indexIn(item->text(nameCol))>=0)
        m.me=true;
    }
  }

  if(m.me) {
    setChildMatchOfParent(item);
    setParentMatchOfChild(item);
  }
}

void FilteredTreeWidget::setChildMatchOfParent(QTreeWidgetItem *item) {
  QTreeWidgetItem *p=item->parent();
  if(!p)
    return;
  Match &m=match[p];
  m.child=true;
  setChildMatchOfParent(p);
}

void FilteredTreeWidget::setParentMatchOfChild(QTreeWidgetItem *item) {
  for(int i=0; i<item->childCount(); i++) {
    QTreeWidgetItem *c=item->child(i);
    Match &m=match[c];
    m.parent=true;
    setParentMatchOfChild(c);
  }
}

void FilteredTreeWidget::updateView(QTreeWidgetItem *item) {
  // recursively walk the tree
  for(int i=0; i<item->childCount(); i++)
    updateView(item->child(i));
  // do nothing but walking the tree for the invisibleRootItem element
  if(item==invisibleRootItem())
    return;

  Match &m=match[item];
  // set hidden
  if(!m.me && !m.parent && !m.child) {
    item->setHidden(true);
    return; // hidden item are not shown so no other flag like color are required to set
  }
  else
    item->setHidden(false);
  // set the color of the column nameCol
  QPalette palette;
  if(m.me) {
    if(!item->isDisabled())
      item->setForeground(nameCol, palette.brush(QPalette::Active, QPalette::Text));
    else
      item->setForeground(nameCol, palette.brush(QPalette::Disabled, QPalette::Text));
  }
  else {
    if(!item->isDisabled())
      item->setForeground(nameCol, QBrush(QColor(255,0,0)));
    else
      item->setForeground(nameCol, QBrush(QColor(128,0,0)));
  }
  // set expanded
  if(m.child)
    item->setExpanded(true);
  else
    item->setExpanded(false);
}

}
