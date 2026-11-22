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

AbstractViewFilter::AbstractViewFilter(QAbstractItemView *view_, int enableRole_) :
  QWidget(view_), view(view_),
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
  connect(view->model(), &QAbstractItemModel::dataChanged, this, [this](const QModelIndex &topLeft, const QModelIndex &bottomRight, const QVector<int> &roles){
    if(filterLE->text().isEmpty())
      return;
    for(auto r : roles)
      if(r != Qt::ForegroundRole) {
        updateItem(topLeft);
        break;
      }
  });
}

void AbstractViewFilter::updateTooltip() {
  QString toolTip = tr("Filter the tree<ul>");
  QString statusTip = tr("Filter by ");
  size_t i=0;
  QString sep;
  for (const auto& col : colMap) {
    if(i<colMap.size()-2)
      sep = ", ";
    else if(i<colMap.size()-1)
      sep = " or ";
    else
      sep = ".";
    i++;
    toolTip += tr("<li>by %1").arg(col.second.second);
    statusTip += col.second.second+sep;
    if(not col.first.isEmpty())
      toolTip += tr(" if the filter starts with \"%1\"").arg(col.first);
    toolTip += sep+tr("</li>");
  }
  toolTip += tr("</ul><dl><dt>Case sensitivity:</dt><dd>%3</dd>"
            "<dt>Filter type:</dt><dd>%4</dd></dl>").arg(caseSensitive?"case sensitive":"case insensitive")
                                                    .arg(filterType==FilterType::RegEx?"regular expression":"glob pattern");
  filterLE->setToolTip(toolTip);
  filterLE->setStatusTip(statusTip);
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
    auto col0 = colMap.find("");
    for (auto col = colMap.begin(); col != colMap.end(); col++) {
      if(col != col0 and filter.pattern().startsWith(col->first)) { // First apply the filter for all non-empty keys 
        const QModelIndex &colIndex=view->model()->index(index.row(), col->second.first, index.parent());
        QRegExp filter2(filter);
        filter2.setPattern(filter.pattern().mid(col->first.size()));
        if(filter2.indexIn(view->model()->data(colIndex, Qt::DisplayRole).value<QString>())>=0)
          m.me=true;
        col0 = colMap.end();
        break;
      }
    }
    if(col0 != colMap.end()) { // Now apply the filter for the empty key if present
      const QModelIndex &colIndex=view->model()->index(index.row(), col0->second.first, index.parent());
      QRegExp filter2(filter);
      filter2.setPattern(filter.pattern().mid(col0->first.size()));
      if(filter2.indexIn(view->model()->data(colIndex, Qt::DisplayRole).value<QString>())>=0)
        m.me=true;
    }
    
    if(m.me) {
      setChildMatchOfParent(index);
      setParentMatchOfChild(index);
    }
  }
  // recursively walk the view
  for(int i=0; i<view->model()->rowCount(index); i++)
    updateMatch(view->model()->index(i, 0, index), filter);
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
    const QModelIndex &c=view->model()->index(i, 0, index);
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
    updateView(view->model()->index(i, 0, index));
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
      updateItem(view->model()->index(i, 0, index),recursive);
  }
}

void AbstractViewFilter::setColumnMap(const std::unordered_map<QString,std::pair<int,QString>> &colMap_) {
  colMap = colMap_;
  updateTooltip();
}

}
