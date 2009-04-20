#include "config.h"
#include "objbody.h"
#include <QFile>
#include <QTextStream>
#include <tinyxml/tinynamespace.h>
#include <Inventor/nodes/SoShapeHints.h>

using namespace std;

double ObjBody::epsVertex=0;

ObjBody::ObjBody(TiXmlElement *element, H5::Group *h5Parent) : RigidBody(element, h5Parent) {
  iconFile=":/objbody.svg";
  setIcon(0, QIcon(iconFile.c_str()));

  // read XML
  TiXmlElement *e=element->FirstChildElement(OPENMBVNS"objFileName");
  string fileName=e->GetText();
  // fix relative path name of file to be included (will hopefully work also on windows)
  fileName=fixPath(e->GetDocument()->ValueStr(), fileName);
  e=e->NextSiblingElement();
  bool textureFromFile=e->GetText()=="true"?true:false;
  e=e->NextSiblingElement();
  bool materialFromFile=e->GetText()=="true"?true:false;
  e=e->NextSiblingElement();
  string normals_=e->GetText();
  Normals normals;
  if(normals_=="fromObjFile") normals=fromObjFile;
  else if(normals_=="flat") normals=flat;
  else if(normals_=="smooth") normals=smooth;
  else normals=smoothIfLessBarrier;
  e=e->NextSiblingElement();
  epsVertex=toVector(e->GetText())[0];
  e=e->NextSiblingElement();
  double epsNormal=toVector(e->GetText())[0];
  e=e->NextSiblingElement();
  double smoothBarrier=toVector(e->GetText())[0];
  e=e->NextSiblingElement();
  string outline_=e->GetText();
  Outline outline;
  if(outline_=="none") outline=none;
  else if(outline_=="calculate") outline=calculate;
  else outline=fromFile;

  // create so
  // read obj
  QFile objFile(fileName.c_str());
  objFile.open(QIODevice::ReadOnly | QIODevice::Text);
  map<QString, ObjGroup> group;
  ObjGroup* curGroup=&group["OpenMBVDefaultGroup"];
  QString dummyStr;
  QRegExp commentRE("[ \t]*#.*");
  QRegExp emptyRE("[ \t]*");
  QRegExp groupRE("[ \t]*g[ \t].*");
  QRegExp vRE("[ \t]*v[ \t].*");
  QRegExp vnRE("[ \t]*vn[ \t].*");
  QRegExp fRE("[ \t]*f[ \t].*");
  //QRegExp fSplitRE("^([0-9]+)/([0-9]*)/([0-9]*)$");
  QRegExp fSplitRE("^([0-9]+)(?:/([0-9]*)(?:/([0-9]*))?)?$");
  while(!objFile.atEnd()) {
    QByteArray line=objFile.readLine();
    line.resize(line.size()-1); // delete "\n"
    if(commentRE.exactMatch(line)) continue; // don't process comment lines
    if(emptyRE.exactMatch(line)) continue; // don't process empty lines
    if(groupRE.exactMatch(line)) { // set current group
      QTextStream stream(line);
      stream>>dummyStr>>dummyStr;
      curGroup=&group[dummyStr];
      continue;
    }
    if(vRE.exactMatch(line)) { // read v (write to current group)
      QTextStream stream(line);
      double x, y, z;
      stream>>dummyStr>>x>>y>>z;
      curGroup->v->point.set1Value(curGroup->v->point.getNum(), x,y,z);
      continue;
    }
    if(vnRE.exactMatch(line)) { // read vn (write to current group)
      QTextStream stream(line);
      double x, y, z;
      stream>>dummyStr>>x>>y>>z;
      curGroup->n->vector.set1Value(curGroup->n->vector.getNum(), x,y,z);
      continue;
    }
    if(fRE.exactMatch(line)) { // read f (write to current group)
      QTextStream stream(line);
      double x, y, z;
      stream>>dummyStr>>dummyStr;
      fSplitRE.indexIn(dummyStr);
      int vi1=fSplitRE.cap(1).toInt()-1, ni1=fSplitRE.cap(3).toInt()-1;
      if(ni1<0) curGroup->hasNormal=false;
      stream>>dummyStr;
      fSplitRE.indexIn(dummyStr);
      int vi2=fSplitRE.cap(1).toInt()-1, ni2=fSplitRE.cap(3).toInt()-1;
      if(ni2<0) curGroup->hasNormal=false;
      while(!stream.atEnd()) {
        stream>>dummyStr;
        if(dummyStr=="") break;
        fSplitRE.indexIn(dummyStr);
        curGroup->f->coordIndex.set1Value(curGroup->f->coordIndex.getNum(), vi1);
        if(curGroup->hasNormal) curGroup->f->normalIndex.set1Value(curGroup->f->normalIndex.getNum(), ni1);
        curGroup->f->coordIndex.set1Value(curGroup->f->coordIndex.getNum(), vi2);
        if(curGroup->hasNormal) curGroup->f->normalIndex.set1Value(curGroup->f->normalIndex.getNum(), ni2);
        vi2=fSplitRE.cap(1).toInt()-1; ni2=fSplitRE.cap(3).toInt()-1;
        if(ni2<0) curGroup->hasNormal=false;
        curGroup->f->coordIndex.set1Value(curGroup->f->coordIndex.getNum(), vi2);
        if(curGroup->hasNormal) curGroup->f->normalIndex.set1Value(curGroup->f->normalIndex.getNum(), ni2);
        curGroup->f->coordIndex.set1Value(curGroup->f->coordIndex.getNum(), -1);
        if(curGroup->hasNormal) curGroup->f->normalIndex.set1Value(curGroup->f->normalIndex.getNum(), -1);
      }
      continue;
    }
  }
  // create so
  map<QString, ObjGroup>::iterator i;
  for(i=group.begin(); i!=group.end(); i++) {
    if(epsVertex>0) { // combine vertex
      SoMFVec3f newvv;
      SoMFInt32 newvi;
      combine(i->second.v->point, newvv, newvi);
      convertIndex(i->second.f->coordIndex, newvi);
      i->second.v->point.copyFrom(newvv);
    }
    soSep->addChild(i->second.v);
    if(i->second.hasNormal) // if hasNormal draw with normals
      soSep->addChild(i->second.n);
    else { // it !hasNormal draw with automatic normals by Inventor and not backface culled and two-sided lighting
      SoShapeHints *sh=new SoShapeHints;
      soSep->addChild(sh);
      sh->vertexOrdering.setValue(SoShapeHints::CLOCKWISE);
      sh->shapeType.setValue(SoShapeHints::UNKNOWN_SHAPE_TYPE);
    }
    soSep->addChild(i->second.f);
  }

  // outline
  //soSep->addChild(soOutLineSwitch);
  //soOutLineSep->addChild(cube);
}

/////////////////////////////
ObjBody::ObjGroup::ObjGroup() {
  v=new SoCoordinate3; v->ref(); v->point.deleteValues(0);
  n=new SoNormal; n->ref(); n->vector.deleteValues(0);
  f=new SoIndexedFaceSet; f->ref(); f->coordIndex.deleteValues(0); f->normalIndex.deleteValues(0);
  hasNormal=true;
}

bool ObjBody::SbVec3fHash::operator()(const SbVec3f& v1, const SbVec3f& v2) const {
  float x1,y1,z1,x2,y2,z2;
  v1.getValue(x1,y1,z1);
  v2.getValue(x2,y2,z2);
  if(x1<x2-epsVertex) return true;
  else if(x1>x2+epsVertex) return false;
  else
    if(y1<y2-epsVertex) return true;
    else if(y1>y2+epsVertex) return false;
    else
      if(z1<z2-epsVertex) return true;
      else if(z1>z2+epsVertex) return false;
      else return false;
};

void ObjBody::combine(const SoMFVec3f& v, SoMFVec3f& newv, SoMFInt32& newvi) {
  map<SbVec3f, int, SbVec3fHash> hash;
  for(int i=0; i<v.getNum(); i++) {
    int &r=hash[*(v.getValues(i))]; // add vertex to a hash map
    // if vertex not exist in hash map copy it to newvv,
    // set corospondenting newvi,
    // and set hash map value to the index of this vertex
    if(r==0) {
      newv.set1Value(newv.getNum(), *v.getValues(i));
      newvi.set1Value(i, newv.getNum()-1);
      r=newv.getNum()-1+1;
    }
    // if vertix exist in hash map,
    // set corrospondenting newvi to the value in the hash map
    else
      newvi.set1Value(i, r-1);
  }
}

void ObjBody::convertIndex(SoMFInt32& fvi, const SoMFInt32& newvi) {
  for(int i=0; i<fvi.getNum(); i++)
    if(fvi[i]>=0) fvi.set1Value(i, newvi[fvi[i]]);
}

// complexibility: ?
bool ObjBody::TwoIndexHash::operator()(const TwoIndex& l1, const TwoIndex& l2) const {
  if(l1.vi1<l2.vi1) return true;
  else if(l1.vi1>l2.vi1) return false;
  else
    if(l1.vi2<l2.vi2) return true;
    else if(l1.vi2>l2.vi2) return false;
    else return false;
}

void ObjBody::computeNormals(const SoMFInt32& fv, const SoMFVec3f &vv, SoMFVec3f& nv) {
//  map<TwoIndex, vector<int>, TwoIndexHash> linem;
//
//  for(int i=0; i<fv.size(); i++) {
//    // set face normals
//    Normal n, t, b;
//    // tangent 1
//    t.x=nv[fv[i].vi2].x-nv[fv[i].vi1].x;
//    t.y=nv[fv[i].vi2].y-nv[fv[i].vi1].y;
//    t.z=nv[fv[i].vi2].z-nv[fv[i].vi1].z;
//    // tagent 2
//    b.x=nv[fv[i].vi3].x-nv[fv[i].vi1].x;
//    b.y=nv[fv[i].vi3].y-nv[fv[i].vi1].y;
//    b.z=nv[fv[i].vi3].z-nv[fv[i].vi1].z;
//    // normal
//    n.x=t.y*b.z-t.z*b.y;
//    n.y=t.z*b.x-t.x*b.z;
//    n.z=t.x*b.y-t.y*b.x;
//    // set normal of all 3 face vertex
//    nv[3*i+0].x=n.x; nv[3*i+0].y=n.y; nv[3*i+0].z=n.z;
//    nv[3*i+1].x=n.x; nv[3*i+1].y=n.y; nv[3*i+1].z=n.z;
//    nv[3*i+2].x=n.x; nv[3*i+2].y=n.y; nv[3*i+2].z=n.z;
//
//    // store all face indexies of each line in a map
//    TwoIndex line;
//    line.vi1=fv[i].vi1; line.vi2=fv[i].vi2;
//    if(line.vi1>line.vi2) { int dummy=line.vi2; line.vi1=line.vi2; line.vi2=dummy; }
//    linem[line].push_back(i);
//    line.vi1=fv[i].vi2; line.vi2=fv[i].vi3;
//    if(line.vi1>line.vi2) { int dummy=line.vi2; line.vi1=line.vi2; line.vi2=dummy; }
//    linem[line].push_back(i);
//    line.vi1=fv[i].vi3; line.vi2=fv[i].vi1;
//    if(line.vi1>line.vi2) { int dummy=line.vi2; line.vi1=line.vi2; line.vi2=dummy; }
//    linem[line].push_back(i);
//  }
//  map<TwoIndex, vector<int>, TwoIndexHash>::iterator i;
//  for(i=linem.begin(); i!=linem.end(); i++) {
//  }
}
/////////////////////////////
