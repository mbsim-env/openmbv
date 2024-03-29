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

#include "config.h"
#include <abstractviewfilter.h>
#include <QGridLayout>
#include <QLabel>
#include <QLineEdit>
#include <QTreeView>
#include <QListView>
#include <QTableView>
#include <QMenu>
#include <utility>

using namespace std;

namespace OpenMBVGUI {

AbstractViewFilter::FilterType AbstractViewFilter::filterType = AbstractViewFilter::FilterType::RegEx;
bool AbstractViewFilter::caseSensitive = false;

void AbstractViewFilter::setFilterType(FilterType filtertype_) {
  filterType=filtertype_;
  staticObject()->optionsChanged();
}

void AbstractViewFilter::setCaseSensitive(bool cs) {
  caseSensitive=cs;
  staticObject()->optionsChanged();
}

AbstractViewFilterStatic* AbstractViewFilter::staticObject() {
  static auto *so=new AbstractViewFilterStatic;
  return so;
}

AbstractViewFilter::AbstractViewFilter(QAbstractItemView *view_, int nameCol_, int typeCol_, QString typePrefix_,
                                       std::function<QObject*(const QModelIndex&)> indexToQObject_, int enableRole_) :
  QWidget(view_), view(view_), nameCol(nameCol_), typeCol(typeCol_), typePrefix(std::move(typePrefix_)), indexToQObject(std::move(indexToQObject_)),
  enableRole(enableRole_) {
  connect(staticObject(), &AbstractViewFilterStatic::optionsChanged, this, [this](){
    updateTooltip();
    applyFilter();
  });

  auto *layout=new QGridLayout(this);
  layout->setContentsMargins(0,0,0,0);
  setLayout(layout);
  layout->addWidget(new QLabel("Filter:"), 0, 0);
  filterLE=new QLineEdit;
  updateTooltip();
  filterLE->setContextMenuPolicy(Qt::CustomContextMenu);
  connect(filterLE, &QAbstractScrollArea::customContextMenuRequested, this, [this](const QPoint &pt){
    auto *menu = filterLE->createStandardContextMenu();

    menu->addSeparator()->setText(tr("Filter type"));
    auto *filterTypeGroup = new QActionGroup(menu);
    auto *regexAction=new QAction("Regular expression filter", filterTypeGroup);
    regexAction->setCheckable(true);
    connect(regexAction, &QAction::triggered, this, [](){ setFilterType(FilterType::RegEx); });
    filterTypeGroup->addAction(regexAction);
    auto *globAction=new QAction("Glob pattern filter", filterTypeGroup);
    globAction->setCheckable(true);
    connect(globAction, &QAction::triggered, this, [](){ setFilterType(FilterType::Glob); });
    filterTypeGroup->addAction(globAction);
    if(filterType==FilterType::RegEx)
      regexAction->setChecked(true);
    else
      globAction->setChecked(true);
    menu->addAction(regexAction);
    menu->addAction(globAction);

    menu->addSeparator()->setText(tr("Case sensitivity"));
    auto *caseGroup = new QActionGroup(menu);
    auto *caseSensAction=new QAction("Case sensitive filter", caseGroup);
    caseSensAction->setCheckable(true);
    connect(caseSensAction, &QAction::triggered, this, [](){ setCaseSensitive(true); });
    caseGroup->addAction(caseSensAction);
    auto *caseInsensAction=new QAction("Case insensitive filter", caseGroup);
    caseInsensAction->setCheckable(true);
    connect(caseInsensAction, &QAction::triggered, this, [](){ setCaseSensitive(false); });
    caseGroup->addAction(caseInsensAction);
    if(caseSensitive)
      caseSensAction->setChecked(true);
    else
      caseInsensAction->setChecked(true);
    menu->addAction(caseSensAction);
    menu->addAction(caseInsensAction);

    menu->exec(filterLE->mapToGlobal(pt));
    delete menu;
  });
  layout->addWidget(filterLE, 0, 1);
  connect(filterLE, &QLineEdit::textEdited, this, &AbstractViewFilter::applyFilter);
  connect(view->model(), &QAbstractItemModel::dataChanged, this, [this](const QModelIndex &index, const QModelIndex &bottomRight){
    if(filterLE->text().isEmpty())
      return;
    updateItem(index);
  });
}

void AbstractViewFilter::updateTooltip() {
  if(typeCol==-2) {
    filterLE->setToolTip(tr("Filter the tree, by applying the given filter on the item names (column %1)."
                           "<dl><dt>Case sensitivity:</dt><dd>%2</dd>"
                           "<dt>Filter type:</dt><dd>%3</dd></dl>").arg(nameCol+1)
                                             .arg(caseSensitive?"case sensitive":"case insensitive")
                                             .arg(filterType==FilterType::RegEx?"regular expression":"glob pattern"));
    filterLE->setStatusTip("Filter by name column.");
  }
  else if(typeCol==-1) {
    filterLE->setToolTip(tr("Filter the tree, by applying the given filter on the item names (column %1)."
                           "Or on the type by :&lt;filter&gt; or on a derived type by ::&lt;filter&gt;."
                           "<dl><dt>Case sensitivity:</dt><dd>%2</dd>"
                           "<dt>Filter type:</dt><dd>%3</dd></dl>").arg(nameCol+1)
                                             .arg(caseSensitive?"case sensitive":"case insensitive")
                                             .arg(filterType==FilterType::RegEx?"regular expression":"glob pattern"));
    filterLE->setStatusTip("Filter by name column, or by type :&lt;filter&gt;, or by derived type ::&lt;filter&gt;");
  }
  else {
    filterLE->setToolTip(tr("Filter the tree, by applying the given filter on the item names (column %1)."
                           "Or on the item type (column %2) if the filter starts with ':'."
                           "<dl><dt>Case sensitivity:</dt><dd>%3</dd>"
                           "<dt>Filter type:</dt><dd>%4</dd></dl>").arg(nameCol+1).arg(typeCol+1)
                                             .arg(caseSensitive?"case sensitive":"case insensitive")
                                             .arg(filterType==FilterType::RegEx?"regular expression":"glob pattern"));
    filterLE->setStatusTip("Filter by name column, or by type column (if the filter starts with ':')");
  }
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
  if(filterLE->text().isEmpty() && oldFilterValue.isEmpty()) {
    matchAll=true;
    return;
  }
  matchAll=false;
  oldFilterValue=filterLE->text();

  QRegExp filter(filterLE->text());
  switch(filterType) {
    case FilterType::RegEx:
      filter.setPatternSyntax(QRegExp::RegExp);
      break;
    case FilterType::Glob:
      filter.setPatternSyntax(QRegExp::Wildcard);
      break;
  }
  filter.setCaseSensitivity(!caseSensitive ? Qt::CaseInsensitive : Qt::CaseSensitive);
  // clear match
  match.clear();
  // updateMatch will fill the variable match
  updateMatch(view->rootIndex(), filter);
  updateView(view->rootIndex());
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
    updateItem(index);
    // set expanded
    auto *tree=qobject_cast<QTreeView*>(view);
    if(tree)
      tree->setExpanded(index, m.child);
  }
  // recursively walk the view
  for(int i=0; i<view->model()->rowCount(index); i++)
    updateView(view->model()->index(i, nameCol, index));
}

void AbstractViewFilter::updateItem(const QModelIndex &index, bool recursive) {
  bool normalColor=true;
  if(!view->model()->flags(index).testFlag(Qt::ItemIsEnabled) ||
     (view->model()->data(index, enableRole).type()==QVariant::Bool && !view->model()->data(index, enableRole).toBool()))
    normalColor=false;
  QPalette palette;
  if(matchAll || match[index].me) {
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
  if(recursive) {
    for(int i=0; i<view->model()->rowCount(index); i++)
      updateItem(view->model()->index(i, nameCol, index),recursive);
  }
}

}
