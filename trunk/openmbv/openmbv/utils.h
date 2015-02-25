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

#ifndef _OPENMBVGUI_UTILS_H_
#define _OPENMBVGUI_UTILS_H_

#include <QtGui/QIcon>
#include <Inventor/nodes/SoSeparator.h>
#include <Inventor/nodes/SoScale.h>
#include <Inventor/SbRotation.h>
#include <string>
#include <Inventor/nodes/SoCoordinate3.h>
#include <Inventor/nodes/SoTriangleStripSet.h>
#include <Inventor/nodes/SoIndexedFaceSet.h>
#include <GL/glu.h>
#include <boost/functional/factory.hpp>
#include <boost/function.hpp>
#include <boost/tuple/tuple.hpp>
#include <openmbvcppinterface/object.h>
#include <QtGui/QTreeWidgetItem>

#ifdef WIN32
#  define CALLMETHOD __stdcall
#else
#  define CALLMETHOD
#endif

namespace OpenMBVGUI {

/** Utilitiy class */
class Utils : virtual public fmatvec::Atom {
  public:
    // INITIALIZATION

    /** initialize the Utils class. Must be called before any member is used. */
    static void initialize();



    // HELPER FUNCTIONS

    /** Use QIconCached(filename) instead of QIcon(filename) everywhere
     * to cache the parsing of e.g. SVG files. This lead to a speedup
     * (at app init) by a factor of 11 in my test case. */
    static const QIcon& QIconCached(std::string filename);

    /** Use SoDBreadAllCached(filename) instead of SoDBreadAll(filename) everywhere
     * to cache the iv-file parsing and scene generation */
    static SoSeparator* SoDBreadAllCached(const std::string &filename);

    /** Convenienc function to draw a frame */
    static SoSeparator* soFrame(double size, double offset, bool pickBBoxAble, SoScale *&scale);
    /** Convenienc function to draw a frame */
    static SoSeparator* soFrame(double size, double offset, bool pickBBoxAble) {
      SoScale *scale; return soFrame(size, offset, pickBBoxAble, scale);
    }

    /** Convenienc function to convert cardan angles to a rotation matrix */
    static SbRotation cardan2Rotation(const SbVec3f& c);
    /** Convenienc function to convert a rotation matrix to cardan angles */
    static SbVec3f rotation2Cardan(const SbRotation& r);

    template<class T>
    static void visitTreeWidgetItems(QTreeWidgetItem *root, boost::function<void (T)> func, bool onlySelected=false);

    static std::string getIconPath();
    static std::string getXMLDocPath();
    static std::string getDocPath();


    // TESSELATION
    static GLUtesselator *tess;


    typedef boost::tuple<QIcon, std::string, boost::function<boost::shared_ptr<OpenMBV::Object>()> > FactoryElement;
    static boost::shared_ptr<OpenMBV::Object> createObjectEditor(const std::vector<FactoryElement> &factory,
                                                                 const std::vector<std::string> &existingNames,
                                                                 const std::string &title);
    template<class T>
    static boost::function<boost::shared_ptr<OpenMBV::Object>()> factory() {
      return static_cast<boost::shared_ptr<T>(*)()>(&OpenMBV::ObjectFactory::create<T>);
    }


  private:
    // INITIALIZATION
    static bool initialized;

    // TESSELATION
    static GLenum tessType;
    static int tessNumVertices;
    static SoTriangleStripSet *tessTriangleStrip;
    static SoIndexedFaceSet *tessTriangleFan;
    static SoCoordinate3 *tessCoord;
    static void CALLMETHOD tessBeginCB(GLenum type, void *data);
    static void CALLMETHOD tessVertexCB(GLdouble *vertex);
    static void CALLMETHOD tessEndCB(void);
};

template<class T>
void Utils::visitTreeWidgetItems(QTreeWidgetItem *root, boost::function<void (T)> func, bool onlySelected) {
  for(int i=0; i<root->childCount(); i++)
    visitTreeWidgetItems(root->child(i), func, onlySelected);
  if((!onlySelected || root->isSelected()) && dynamic_cast<T>(root))
    func(static_cast<T>(root));
}

}

#endif
