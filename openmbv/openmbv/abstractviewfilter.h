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

#ifndef _OPENMBVGUI_ABSTRACTVIEWFILTER_H_
#define _OPENMBVGUI_ABSTRACTVIEWFILTER_H_

#include <QWidget>
#include <QAbstractItemView>
#include <functional>

// If Coin and SoQt is linked as a dll no symbols of this file are exported (for an unknown reason).
// Hence we explicitly export ALL symbols.
// We cannot export selectively symbols since some symbols defined by Q_OBJECT needs also to be exported.
// The clear Qt way would be to use PImpl but this is not done here.
#ifdef _WIN32
#  define DLL_PUBLIC __declspec(dllexport)
#else
#  define DLL_PUBLIC
#endif

namespace OpenMBVGUI {

class DLL_PUBLIC AbstractViewFilterStatic : public QObject {
  Q_OBJECT
  Q_SIGNALS:
    void optionsChanged();
};

/*! A filter for QTreeView classes (like QTreeWidget) */
class DLL_PUBLIC AbstractViewFilter : public QWidget {
  public:
    /*! Creates a filter for QTreeView.
     * To column to which the filter is applied is defined by a string prefix of the filter and
     * setColumnMap which maps prefixes to column number.
     * The filter can use glob or regex matches and can be case sensitive or insensitive.
     * Coloring of matching items is done using the setData function of the view-model using the Qt::ForegroundRole. Hence,
     * to enable filtered coloring the data function of the view-model should directly return the QBrush set using setData.
     * The boolean view-model flag enableRole_ is honored when coloring is done be the filter:
     * If this flag is true normal coloring is done if false "disabled" coloring is done. */
    AbstractViewFilter(QAbstractItemView *view_, int enableRole_=Qt::UserRole);

    //! Set the filter programatically.
    //! Setting the filter applies the filter on the view.
    void setFilter(const QString &filter);

    //! Applies the current filter on the view.
    //! This is automatically done when using setFilter.
    void applyFilter();

    enum class FilterType { RegEx, Glob };
    static void setFilterType(FilterType filtertype_);
    static void setCaseSensitive(bool cs);
    static FilterType getFilterType() { return filterType; }
    static bool getCaseSensitive() { return caseSensitive; }
    static AbstractViewFilterStatic* staticObject();
    void updateItem(const QModelIndex &index, bool recursive=false);
    void setColumnMap(const std::unordered_map<QString,std::pair<int,QString>> &colMap_);

  protected:
    void updateTooltip();

    // update the match varaible
    void updateMatch(const QModelIndex &index, const QRegExp &filter);
    void setChildMatchOfParent(const QModelIndex &index);
    void setParentMatchOfChild(const QModelIndex &index);

    // update the view using the current match variable
    void updateView(const QModelIndex &index);

    static FilterType filterType;
    static bool caseSensitive;

    QLineEdit *filterLE;
    QString oldFilterValue;
    QAbstractItemView *view;
    bool matchAll { true };
    int enableRole;
    std::unordered_map<QString,std::pair<int,QString>> colMap;

    struct Match {
      Match()  = default;
      bool parent{false}; // any parent matches
      bool me{false};     // myself matches
      bool child{false};  // any child matches
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
