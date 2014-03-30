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

/*! A QTreeWidget including a filter for tree items */
class FilteredTreeWidget : public QTreeWidget {
  Q_OBJECT;
  public:
    /*! Creates a filtered QTreeWidget.
     * \p nameCol_ defines the column against normal regex searches (<regex>) are made.
     * If \p typeCol_ >= 0 it is the column against type serches (:<typename>) are made.
     * If \p typeCol_ = -1 type searches (:<typename> or ::<typename>) are made against the qt-metaobject type of the items.
     * If \p typeCol_ = -2 type searches are disabled, everything is treated as a normal regex serach.
     * The typename of a type search (:<typename> or ::<typename>) is prefixes with \p typePrefix_. Usually
     * this is a namespace e.g. 'OpenMBVGUI::'. */
    FilteredTreeWidget(QWidget *parent=NULL, int nameCol_=0, int typeCol_=-2, const QString &typePrefix_="");

    //! Set the filter programatically.
    //! Setting the filter applies the filter on the tree.
    void setFilter(const QString &filter);

    //! Get the filter widget consisting of the label "Filter:" and a QTextEdit.
    QWidget *getFilterWidget() { return widget; }

  public slots:
    //! Applies the current filter on the tree.
    //! This is automatically done when using setFilter or when pressing enter in the filter QTextEdit.
    void applyFilter();

  protected:
    // update the match varaible
    void updateMatch(QTreeWidgetItem *item, const QRegExp &filter);
    void setChildMatchOfParent(QTreeWidgetItem *item);
    void setParentMatchOfChild(QTreeWidgetItem *item);

    // update the tree view using the current match variable
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
