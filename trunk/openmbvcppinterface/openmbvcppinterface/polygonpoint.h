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

/*
   OpenMBVC++Interface - C++ Interface for Body- and Pos-File Creation
   Used by OpenMBV
   Copyright (C) 2006 Institute of Applied Mechanics,
   Technical University of Munich

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
   Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
   */

#ifndef POLYGONPOINT_H
#define POLYGONPOINT_H

namespace OpenMBV {

  //! Polygon point
  /*!
   * x and y are the coordinates of a polygon-edge. If b is 0 this
   * edge is in reality not a edge and is rendered smooth in OpenMBV. If b is 1
   * this edge is rendered non-smooth in OpenMBV.
   */
  struct PolygonPoint {
    float x, y;
    int b;
  };

}

#endif
