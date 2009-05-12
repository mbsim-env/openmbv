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
#include "objbody.h"
#include <QFile>
#include <QTextStream>
#include <tinynamespace.h>
#include <Inventor/nodes/SoShapeHints.h>
#include <Inventor/actions/SoGetBoundingBoxAction.h>

using namespace std;

ObjBody::ObjBody(TiXmlElement *element, H5::Group *h5Parent, QTreeWidgetItem *parentItem, SoGroup *soParent) : RigidBody(element, h5Parent, parentItem, soParent) {
  iconFile=":/objbody.svg";
  setIcon(0, QIcon(iconFile.c_str()));

  // read XML
  TiXmlElement *e=element->FirstChildElement(OPENMBVNS"objFileName");
  string fileName=e->GetText();
  // fix relative path name of file to be included (will hopefully work also on windows)
  fileName=fixPath(e->GetDocument()->ValueStr(), fileName);
  e=e->NextSiblingElement();
  bool textureFromFile=(e->GetText()==string("true"))?true:false;
  e=e->NextSiblingElement();
  bool materialFromFile=(e->GetText()==string("true"))?true:false;
  e=e->NextSiblingElement();
  string normals_=e->GetText();
  Normals normals;
  if(normals_=="fromObjFile") normals=fromObjFile;
  else if(normals_=="flat") normals=flat;
  else if(normals_=="smooth") normals=smooth;
  else normals=smoothIfLessBarrier;
  e=e->NextSiblingElement();
  double epsVertex=toVector(e->GetText())[0];
  e=e->NextSiblingElement();
  double epsNormal=toVector(e->GetText())[0];
  e=e->NextSiblingElement();
  smoothBarrier=toVector(e->GetText())[0];
  e=e->NextSiblingElement();
  string outline_=e->GetText();
  Outline outline;
  if(outline_=="none") outline=none;
  else if(outline_=="calculate") outline=calculate;
  else outline=fromFile;

  // create so
  // read obj
  v=new SoCoordinate3; v->point.deleteValues(0);
  t=new SoTextureCoordinate2; t->point.deleteValues(0);
  n=new SoNormal; n->vector.deleteValues(0);
  map<QString, SoMaterial*> material;
  map<QString, SoTexture2*> map_;
  QFile objFile(fileName.c_str());
  objFile.open(QIODevice::ReadOnly | QIODevice::Text);
  map<QString, MtlMapGroup> group;
  MtlMapGroup* curGroup=&group["OpenMBVDefaultMtlMap"];
  QString dummyStr;
  QRegExp commentRE("[ \t]*#.*");
  QRegExp emptyRE("[ \t]*");
  QRegExp usemtlRE("[ \t]*usemtl[ \t].*");
  QRegExp mtllibRE("[ \t]*mtllib[ \t].*");
  QRegExp usemapRE("[ \t]*usemap[ \t].*");
  QRegExp maplibRE("[ \t]*maplib[ \t].*");
  QRegExp vRE("[ \t]*v[ \t].*");
  QRegExp vtRE("[ \t]*vt[ \t].*");
  QRegExp vnRE("[ \t]*vn[ \t].*");
  QRegExp fRE("[ \t]*f[ \t].*");
  QRegExp fSplitRE("^([0-9]+)(?:/([0-9]*)(?:/([0-9]*))?)?$");
  QRegExp lRE("[ \t]*l[ \t].*");
  QRegExp lSplitRE("^([0-9]+)(?:/([0-9]*))?$");
  while(!objFile.atEnd()) {
    QByteArray line=objFile.readLine();
    line.resize(line.size()-1); // delete "\n"
    if(commentRE.exactMatch(line)) continue; // don't process comment lines
    if(emptyRE.exactMatch(line)) continue; // don't process empty lines
    if(materialFromFile && mtllibRE.exactMatch(line)) { // read material
      QTextStream stream(line);
      stream>>dummyStr>>dummyStr;
      string mtlFile=fixPath(fileName, dummyStr.toStdString());
      readMtlLib(mtlFile, material);
      continue;
    }
    if(textureFromFile && maplibRE.exactMatch(line)) { // read map
      QTextStream stream(line);
      stream>>dummyStr>>dummyStr;
      string mapFile=fixPath(fileName, dummyStr.toStdString());
      readMapLib(mapFile, map_);
      continue;
    }
    if(usemtlRE.exactMatch(line)) { // set current group
      QTextStream stream(line);
      stream>>dummyStr>>dummyStr;
      SoTexture2 *mapSave=curGroup->map; // save map
      curGroup=&group[dummyStr];
      if(materialFromFile) curGroup->mat=material[dummyStr];
      curGroup->map=mapSave; // set saved map
      continue;
    }
    if(usemapRE.exactMatch(line)) { // set current group
      QTextStream stream(line);
      stream>>dummyStr>>dummyStr;
      SoMaterial *matSave=curGroup->mat; // save mat
      curGroup=&group[dummyStr];
      if(textureFromFile) curGroup->map=map_[dummyStr];
      curGroup->mat=matSave; // set saved mat
      continue;
    }
    if(vRE.exactMatch(line)) { // read v
      QTextStream stream(line);
      double x, y, z;
      stream>>dummyStr>>x>>y>>z;
      v->point.set1Value(v->point.getNum(), x,y,z);
      continue;
    }
    if(textureFromFile && vtRE.exactMatch(line)) { // read vt
      QTextStream stream(line);
      double x, y;
      stream>>dummyStr>>x>>y;
      t->point.set1Value(t->point.getNum(), x,y);
      continue;
    }
    if(vnRE.exactMatch(line)) { // read vn
      QTextStream stream(line);
      double x, y, z;
      stream>>dummyStr>>x>>y>>z;
      n->vector.set1Value(n->vector.getNum(), x,y,z);
      continue;
    }
    if(fRE.exactMatch(line)) { // read f (write to current group)
      QTextStream stream(line);
      stream>>dummyStr>>dummyStr;
      fSplitRE.indexIn(dummyStr);
      int vi1=fSplitRE.cap(1).toInt()-1, ni1=fSplitRE.cap(3).toInt()-1;
      stream>>dummyStr;
      fSplitRE.indexIn(dummyStr);
      int vi2=fSplitRE.cap(1).toInt()-1, ni2=fSplitRE.cap(3).toInt()-1;
      while(!stream.atEnd()) {
        stream>>dummyStr;
        if(dummyStr=="") break;
        fSplitRE.indexIn(dummyStr);
        curGroup->f->coordIndex.set1Value(curGroup->f->coordIndex.getNum(), vi1);
        curGroup->f->normalIndex.set1Value(curGroup->f->normalIndex.getNum(), ni1);
        curGroup->f->coordIndex.set1Value(curGroup->f->coordIndex.getNum(), vi2);
        curGroup->f->normalIndex.set1Value(curGroup->f->normalIndex.getNum(), ni2);
        vi2=fSplitRE.cap(1).toInt()-1; ni2=fSplitRE.cap(3).toInt()-1;
        curGroup->f->coordIndex.set1Value(curGroup->f->coordIndex.getNum(), vi2);
        curGroup->f->normalIndex.set1Value(curGroup->f->normalIndex.getNum(), ni2);
        curGroup->f->coordIndex.set1Value(curGroup->f->coordIndex.getNum(), -1);
        curGroup->f->normalIndex.set1Value(curGroup->f->normalIndex.getNum(), -1);
      }
      continue;
    }
    if(lRE.exactMatch(line)) { // read l (write to current group)
      QTextStream stream(line);
      stream>>dummyStr;
      while(!stream.atEnd()) {
        stream>>dummyStr;
        if(dummyStr=="") break;
        lSplitRE.indexIn(dummyStr);
        curGroup->ol->coordIndex.set1Value(curGroup->ol->coordIndex.getNum(), lSplitRE.cap(1).toInt()-1);
      }
      curGroup->ol->coordIndex.set1Value(curGroup->ol->coordIndex.getNum(), -1);
      continue;
    }
  }

  // create so
  map<QString, MtlMapGroup>::iterator i;
  // vertex
  if(epsVertex>0) { // combine vertex
   SoMFVec3f newvv;
   SoMFInt32 newvi;
   eps=epsVertex;
   combine(v->point, newvv, newvi);
   v->point.copyFrom(newvv);
   for(i=group.begin(); i!=group.end(); i++)
     convertIndex(i->second.f->coordIndex, newvi);
  }
  soSep->addChild(v);
  // texture points
  if(textureFromFile) soSep->addChild(t);
  // no backface culling; two side lithning
  SoShapeHints *sh=new SoShapeHints;
  soSep->addChild(sh);
  sh->vertexOrdering.setValue(SoShapeHints::UNKNOWN_ORDERING);
  sh->shapeType.setValue(SoShapeHints::UNKNOWN_SHAPE_TYPE);
  // normals and shape
  if(normals==flat) { // flat normals
    SoNormalBinding *nb=new SoNormalBinding;
    soSep->addChild(nb);
    nb->value.setValue(SoNormalBinding::PER_FACE);
    for(i=group.begin(); i!=group.end(); i++) {
      if(i->second.mat) soSep->addChild(i->second.mat);
      if(i->second.map) soSep->addChild(i->second.map);
      soSep->addChild(i->second.f);
    }
  }
  else if(normals==fromObjFile) { // use normals form obj
    if(epsNormal>0) { // combine normals
     SoMFVec3f newvv;
     SoMFInt32 newvi;
     eps=epsNormal;
     combine(n->vector, newvv, newvi);
     n->vector.copyFrom(newvv);
     for(i=group.begin(); i!=group.end(); i++)
       convertIndex(i->second.f->normalIndex, newvi);
    }
    soSep->addChild(n);
    for(i=group.begin(); i!=group.end(); i++) {
      if(i->second.mat) soSep->addChild(i->second.mat);
      if(i->second.map) soSep->addChild(i->second.map);
      soSep->addChild(i->second.f);
    }
  }
  else { // smooth or real normals
    if(normals==smooth) smoothBarrier=M_PI; // if smooth simply use high value for barrier
    for(i=group.begin(); i!=group.end(); i++) {
      SoMFInt32 fni;
      SoMFVec3f n;
      SoMFInt32 oli;
      computeNormals(i->second.f->coordIndex, v->point, fni, n, oli, smoothBarrier);
      i->second.f->normalIndex.copyFrom(fni);
      if(epsNormal>0) { // combine normals
       SoMFVec3f newvv;
       SoMFInt32 newvi;
       eps=epsNormal;
       combine(n, newvv, newvi);
       i->second.n->vector.copyFrom(newvv);
       convertIndex(i->second.f->normalIndex, newvi);
      }
      soSep->addChild(i->second.n);
      if(i->second.mat) soSep->addChild(i->second.mat);
      if(i->second.map) soSep->addChild(i->second.map);
      soSep->addChild(i->second.f);
      if(outline==calculate) {
        i->second.ol->coordIndex.copyFrom(oli);
        soOutLineSep->addChild(i->second.ol);
      }
    }
  }
  if(outline==fromFile)
    for(i=group.begin(); i!=group.end(); i++)
      soOutLineSep->addChild(i->second.ol);
  soSep->addChild(soOutLineSwitch);

  // scale ref/localFrame
  SoGetBoundingBoxAction bboxAction(SbViewportRegion(0,0));
  bboxAction.apply(soSep);
  float x1,y1,z1,x2,y2,z2;
  bboxAction.getBoundingBox().getBounds(x1,y1,z1,x2,y2,z2);
  double size=min(x2-x1,min(y2-y1,z2-z1));
  refFrameScale->scaleFactor.setValue(size,size,size);
  localFrameScale->scaleFactor.setValue(size,size,size);
}

void ObjBody::readMtlLib(const std::string& mtlFile_, std::map<QString, SoMaterial*>& material) {
  QFile mtlFile(mtlFile_.c_str());
  mtlFile.open(QIODevice::ReadOnly | QIODevice::Text);
  QString dummyStr;
  QRegExp commentRE("[ \t]*#.*");
  QRegExp emptyRE("[ \t]*");
  QRegExp newmtlRE("[ \t]*newmtl[ \t].*");
  QRegExp KaRE("[ \t]*Ka[ \t].*");
  QRegExp KdRE("[ \t]*Kd[ \t].*");
  QRegExp KsRE("[ \t]*Ks[ \t].*");
  QRegExp NsRE("[ \t]*Ns[ \t].*");
  QRegExp dRE("[ \t]*d[ \t].*");
  SoMaterial *curMaterial;
  while(!mtlFile.atEnd()) {
    QByteArray line=mtlFile.readLine();
    line.resize(line.size()-1); // delete "\n"
    if(commentRE.exactMatch(line)) continue; // don't process comment lines
    if(emptyRE.exactMatch(line)) continue; // don't process empty lines
    if(newmtlRE.exactMatch(line)) { // new material
      QTextStream stream(line);
      stream>>dummyStr>>dummyStr;
      curMaterial=new SoMaterial;
      material[dummyStr]=curMaterial;
      continue;
    }
    if(KaRE.exactMatch(line)) { // set Ka
      QTextStream stream(line);
      double r, g, b;
      stream>>dummyStr>>r>>g>>b;
      curMaterial->ambientColor.setValue(r,g,b);
      continue;
    }
    if(KdRE.exactMatch(line)) { // set Kd
      QTextStream stream(line);
      double r, g, b;
      stream>>dummyStr>>r>>g>>b;
      curMaterial->diffuseColor.setValue(r,g,b);
      continue;
    }
    if(KsRE.exactMatch(line)) { // set Ks
      QTextStream stream(line);
      double r, g, b;
      stream>>dummyStr>>r>>g>>b;
      curMaterial->specularColor.setValue(r,g,b);
      continue;
    }
    if(NsRE.exactMatch(line)) { // set Ns
      QTextStream stream(line);
      double s;
      stream>>dummyStr>>s;
      curMaterial->shininess.setValue(s/1000);
      continue;
    }
    if(dRE.exactMatch(line)) { // set d
      QTextStream stream(line);
      double s;
      stream>>dummyStr>>s;
      curMaterial->transparency.setValue(1-s);
      continue;
    }
  }
}

void ObjBody::readMapLib(const std::string& mapFile_, std::map<QString, SoTexture2*>& map_) {
  QFile mapFile(mapFile_.c_str());
  mapFile.open(QIODevice::ReadOnly | QIODevice::Text);
  QString dummyStr;
  QRegExp commentRE("[ \t]*#.*");
  QRegExp emptyRE("[ \t]*");
  QRegExp newmapRE("[ \t]*newmap[ \t].*");
  QRegExp KdRE("[ \t]*Kd[ \t].*");
  SoTexture2 *curMap;
  while(!mapFile.atEnd()) {
    QByteArray line=mapFile.readLine();
    line.resize(line.size()-1); // delete "\n"
    if(commentRE.exactMatch(line)) continue; // don't process comment lines
    if(emptyRE.exactMatch(line)) continue; // don't process empty lines
    if(newmapRE.exactMatch(line)) { // new map
      QTextStream stream(line);
      stream>>dummyStr>>dummyStr;
      curMap=new SoTexture2;
      map_[dummyStr]=curMap;
      continue;
    }
    if(KdRE.exactMatch(line)) { // set Kd
      QTextStream stream(line);
      stream>>dummyStr>>dummyStr;
      string imgName=fixPath(mapFile_, dummyStr.toStdString());
      QImage img(imgName.c_str());
      unsigned char *buf=new unsigned char[img.width()*img.height()*4];
      for(int y=0; y<img.height(); y++)
        for(int x=0; x<img.width(); x++) {
          int o=((img.height()-y-1)*img.width()+x)*4;
          QRgb rgba=img.pixel(x,y);
          buf[o+0]=qRed(rgba);
          buf[o+1]=qGreen(rgba);
          buf[o+2]=qBlue(rgba);
          buf[o+3]=qAlpha(rgba);
        }
      curMap->image.setValue(SbVec2s(img.height(),img.width()),4,buf);
      delete[]buf;
      continue;
    }
  }
}

ObjBody::MtlMapGroup::MtlMapGroup() {
  f=new SoIndexedFaceSet; f->ref(); f->coordIndex.deleteValues(0); f->normalIndex.deleteValues(0);
  ol=new SoIndexedLineSet; ol->ref(); ol->coordIndex.deleteValues(0);
  n=new SoNormal; n->ref(); n->vector.deleteValues(0);
  mat=0;
  map=0;
}
