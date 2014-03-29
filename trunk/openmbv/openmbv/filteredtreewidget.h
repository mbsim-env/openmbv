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

#ifndef _OPENMBVGUI_FILTEREDTREEWIDGET_H_
#define _OPENMBVGUI_FILTEREDTREEWIDGET_H_

#include <QtGui/QTreeWidget>

namespace OpenMBVGUI {

class FilteredTreeWidget : public QTreeWidget {
  Q_OBJECT;
  public:
    FilteredTreeWidget(QWidget *parent=NULL, const QString &typePrefix_="", int nameCol_=0, int typeCol_=-1);
    void setFilter(const QString &filter);
    QWidget *getFilterWidget() { return widget; }
  public slots:
    void updateFilter();
  protected:
    void updateMatch(QTreeWidgetItem *item, const QRegExp &filter);
    void setChildMatchOfParent(QTreeWidgetItem *item);
    void setParentMatchOfChild(QTreeWidgetItem *item);
    void updateView(QTreeWidgetItem *item);
    QWidget *widget;
    QLineEdit *filterLE;
    QString typePrefix;
    int nameCol;
    int typeCol;
    struct Match {
      Match() : parent(false), me(false), child(false) {}
      bool parent; // any parent matches
      bool me;     // myself matches
      bool child;  // any child matches
    };
    std::map<QTreeWidgetItem*, Match> match;
};

}

#endif
