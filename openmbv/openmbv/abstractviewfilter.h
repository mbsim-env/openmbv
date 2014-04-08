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

#ifndef _OPENMBVGUI_ABSTRACTVIEWFILTER_H_
#define _OPENMBVGUI_ABSTRACTVIEWFILTER_H_

#include <QtGui/QWidget>
#include <QtGui/QAbstractItemView>
#include <boost/function.hpp>

// If Coin and SoQt is linked as a dll no symbols of this file are exported (for an unknown reason).
// Hence we explicitly export the required symbols. This should be done for all code for a clean Windows build!
#ifdef _WIN32
#  define DLL_PUBLIC __declspec(dllexport)
#else
#  define DLL_PUBLIC
#endif

namespace OpenMBVGUI {

/*! A filter for QTreeView classes (like QTreeWidget) */
class AbstractViewFilter : public QWidget {
  Q_OBJECT;
  public:
    /*! Creates a filter for QTreeView.
     * \p nameCol_ defines the column against normal regex searches (<regex>) are made (Qt::DisplayRole).
     * If \p typeCol_ >= 0 it is the column against type searches (:<typename>) are made (Qt::DisplayRole).
     * If \p typeCol_ = -1 type searches (:<typename> or ::<typename>) are made against the qt-metaobject type of the items.
     * In this case \a indexToQObject_ must be provided and convert a given model index to the corresponding QObject
     * (normally using internalPointer of the index).
     * If \p typeCol_ = -2 type searches are disabled, everything is treated as a normal regex serach on nameCol_.
     * The typename of a type search (:<typename> or ::<typename>) is prefixes with \p typePrefix_. Usually
     * this is a namespace e.g. 'OpenMBVGUI::'.
     * Coloring of matching items is done using the setData function of the view-model using the Qt::ForegroundRole. Hence,
     * to enable filtered coloring the data function of the view-model should directly return the QBrush set using setData.
     * The view-model flag Qt::ItemIsEnabled is honored by the filtered coloring. */
    DLL_PUBLIC AbstractViewFilter(QAbstractItemView *view_, int nameCol_=0, int typeCol_=-2, const QString &typePrefix_="",
                       boost::function<QObject*(const QModelIndex&)> indexToQObject_=boost::function<QObject*(const QModelIndex&)>(),
                       int enableRole_=Qt::UserRole);

    //! Set the filter programatically.
    //! Setting the filter applies the filter on the view.
    DLL_PUBLIC void setFilter(const QString &filter);

  public slots:
    //! Applies the current filter on the view.
    //! This is automatically done when using setFilter or when pressing enter in the filter QTextEdit.
    DLL_PUBLIC void applyFilter();

  protected:
    // update the match varaible
    void updateMatch(const QModelIndex &index, const QRegExp &filter);
    void setChildMatchOfParent(const QModelIndex &index);
    void setParentMatchOfChild(const QModelIndex &index);

    // update the view using the current match variable
    void updateView(const QModelIndex &index);

    QLineEdit *filterLE;
    QAbstractItemView *view;
    int nameCol;
    int typeCol;
    QString typePrefix;
    boost::function<QObject*(const QModelIndex&)> indexToQObject;
    int enableRole;

    struct Match {
      Match() : parent(false), me(false), child(false) {}
      bool parent; // any parent matches
      bool me;     // myself matches
      bool child;  // any child matches
    };
    std::map<QModelIndex, Match> match;

    template<typename View>
    bool setRowHidden2(View *view, const Match &m, const QModelIndex &index);
    template<typename View>
    bool setRowHidden3(View *view, const Match &m, const QModelIndex &index);
};

template<typename View>
bool AbstractViewFilter::setRowHidden2(View *view, const Match &m, const QModelIndex &index) {
  if(!view)
    return false;
  if(!m.me && !m.parent && !m.child) {
    view->setRowHidden(index.row(), true);
    return true; // hidden index are not shown so no other flag like color are required to set
  }
  else
    view->setRowHidden(index.row(), false);
  return false;
}

template<typename View>
bool AbstractViewFilter::setRowHidden3(View *view, const Match &m, const QModelIndex &index) {
  if(!view)
    return false;
  if(!m.me && !m.parent && !m.child) {
    view->setRowHidden(index.row(), index.parent(), true);
    return true; // hidden index are not shown so no other flag like color are required to set
  }
  else
    view->setRowHidden(index.row(), index.parent(), false);
  return false;
}

}

#endif
