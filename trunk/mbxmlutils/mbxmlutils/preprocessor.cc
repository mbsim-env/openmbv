#include <libxml/xmlschemas.h>
#include <iostream>
#include <fstream>
extern "C" {
#include <regex.h>
}
#include "env.h"
#include "mbxmlutilstinyxml/tinyxml-src/tinyxml.h"
#include "mbxmlutilstinyxml/tinyxml-src/tinynamespace.h"

#define MBXMLUTILSPVNS "{http://openmbv.berlios.de/MBXMLUtils/physicalvariable}"

using namespace std;

char *nslocation;

// validate file using schema (currently by libxml)
int validate(const char *schema, const char *file) {
  xmlDoc *doc;
  cout<<"Parse and validate "<<file<<endl;
  doc=xmlParseFile(file);
  if(!doc) { cout<<"ERROR parsing "<<file<<endl; return 1; }
  int ret=xmlSchemaValidateDoc(xmlSchemaNewValidCtxt(xmlSchemaParse(xmlSchemaNewParserCtxt(schema))), doc);
  if(ret!=0) { cout<<"ERROR validating "<<file<<endl; return ret; }
  xmlFreeDoc(doc);
  return 0;
}

int toOctave(TiXmlElement *&e) {
  if(e->ValueStr()==MBXMLUTILSPVNS"xmlMatrix") {
    string mat="[";
    for(TiXmlElement* row=e->FirstChildElement(); row!=0; row=row->NextSiblingElement()) {
      for(TiXmlElement* ele=row->FirstChildElement(); ele!=0; ele=ele->NextSiblingElement()) {
        mat=mat+ele->GetText();
        if(ele->NextSiblingElement()) mat=mat+",";
      }
      if(row->NextSiblingElement()) mat=mat+";";
    }
    mat=mat+"]";
    TiXmlText *text=new TiXmlText(mat);
    e->Parent()->InsertEndChild(*text);
    e->Parent()->RemoveChild(e);
    e=0;
    return 0;
  }
  if(e->ValueStr()==MBXMLUTILSPVNS"xmlVector") {
    string vec="[";
    for(TiXmlElement* ele=e->FirstChildElement(); ele!=0; ele=ele->NextSiblingElement()) {
      vec=vec+ele->GetText();
      if(ele->NextSiblingElement()) vec=vec+";";
    }
    vec=vec+"]";
    TiXmlText *text=new TiXmlText(vec);
    e->Parent()->InsertEndChild(*text);
    e->Parent()->RemoveChild(e);
    e=0;
    return 0;
  }
  if(e->ValueStr()==MBXMLUTILSPVNS"asciiVectorRef" || e->ValueStr()==MBXMLUTILSPVNS"asciiMatrixRef") {
    ifstream file(e->Attribute("href"));
    if(file.fail()) { cout<<"ERROR! Can not open file: "<<e->Attribute("href")<<endl; return 1; }
    string line, vec;
    while(!file.eof()) {;
      getline(file, line);
      // delete comments starting with % or # and append lines to vec separated by ;
      size_t pos=line.find_first_of("%#");
      if(pos!=string::npos) vec=vec+line.substr(0,pos)+";"; else vec=vec+line+";";
    }
    regex_t re;
    regmatch_t pmatch[1];
    // replace sequences of ; with ;
    regcomp(&re, ";( *;)+", REG_EXTENDED);
    while(regexec(&re, vec.c_str(), 1, pmatch, 0)==0)
      vec=vec.substr(0, pmatch[0].rm_so)+";"+vec.substr(pmatch[0].rm_eo);
    regfree(&re);
    // delete leading ;
    regcomp(&re, "^ *;", REG_EXTENDED);
    if(regexec(&re, vec.c_str(), 1, pmatch, 0)==0)
      vec=vec.substr(pmatch[0].rm_eo);
    regfree(&re);
    // delete tailing ;
    regcomp(&re, "; *$", REG_EXTENDED);
    if(regexec(&re, vec.c_str(), 1, pmatch, 0)==0)
      vec=vec.substr(0,pmatch[0].rm_so);
    regfree(&re);
    vec="["+vec+"]";
    TiXmlText *text=new TiXmlText(vec);
    e->Parent()->InsertEndChild(*text);
    e->Parent()->RemoveChild(e);
    e=0;
    return 0;
  }

  TiXmlElement *c=e->FirstChildElement();
  while(c) {
    if(toOctave(c)!=0) return 1;
    if(c==0) break; // break if c was removed by toOctave at the line below
    c=c->NextSiblingElement();
  }

  return 0;
}

string replace(string str, string search, string subst) {
  regmatch_t pmatch[3];
  regex_t re1, re2, re3, re4;
  regcomp(&re1, ("^(.*[^a-zA-Z_])"+search+"([^a-zA-Z0-9_].*)$").c_str(), REG_EXTENDED);
  regcomp(&re2, ("^"+search+"([^a-zA-Z0-9_].*)$").c_str(), REG_EXTENDED);
  regcomp(&re3, ("^(.*[^a-zA-Z_])"+search+"$").c_str(), REG_EXTENDED);
  regcomp(&re4, ("^"+search+"$").c_str(), REG_EXTENDED);
  bool break_=false;
  do {
    if(regexec(&re1, str.c_str(), 3, pmatch, 0)==0)
      str=str.substr(pmatch[1].rm_so, pmatch[1].rm_eo-pmatch[1].rm_so)+
                    subst+str.substr(pmatch[2].rm_so, pmatch[2].rm_eo-pmatch[2].rm_so);
    else if(regexec(&re2, str.c_str(), 3, pmatch, 0)==0)
      str=subst+str.substr(pmatch[1].rm_so, pmatch[1].rm_eo-pmatch[1].rm_so);
    else if(regexec(&re3, str.c_str(), 3, pmatch, 0)==0)
      str=str.substr(pmatch[1].rm_so, pmatch[1].rm_eo-pmatch[1].rm_so)+subst;
    else if(regexec(&re4, str.c_str(), 3, pmatch, 0)==0)
      str=subst;
    else break_=true;
  } while(break_==false);
  regfree(&re1);
  regfree(&re2);
  regfree(&re3);
  regfree(&re4);
  return str;
}
void resubst(TiXmlElement *e, const char *counterName, const char *subst) {
  // resubst all text elements
  if(e->GetText())
    e->FirstChild()->SetValue(replace(e->GetText(), counterName, subst));

  // resubst name and ref* attributes elements
  TiXmlAttribute *a=e->FirstAttribute();
  for(TiXmlAttribute *a=e->FirstAttribute(); a!=0; a=a->Next()) {
    if(a->Name()==string("name") ||
       string(a->Name()).substr(0,3)=="ref")
      a->SetValue(replace(a->ValueStr(), counterName, subst));
  }

  TiXmlElement *c=e->FirstChildElement();
  while(c) {
    resubst(c, counterName, subst);
    c=c->NextSiblingElement();
  }
}

int embed(TiXmlElement *&e, map<string,string> &nsprefix, TiXmlElement *paramxmlroot) {
  if(e->ValueStr()==MBXMLUTILSPVNS"embed") {
    // check
    if((e->Attribute("href") && e->FirstChildElement()) ||
       (e->Attribute("href")==0 && e->FirstChildElement()==0)) {
      cout<<"ERROR! Only the href attribute OR a child element is allowed in embed!"<<endl;
      return 1;
    }

    // file name if href attribute exist
    string file="";
    if(e->Attribute("href"))
      file=e->Attribute("href");

    // subst count by parameter
    string countstr=e->Attribute("count");
    if(paramxmlroot)
      for(TiXmlElement *el=paramxmlroot->FirstChildElement(); el!=0; el=el->NextSiblingElement())
        countstr=replace(countstr, el->Attribute("name"), el->GetText());
    int count=atoi(countstr.c_str());

    // couter name
    string counterName=e->Attribute("counterName");

    // validate/load if file is given
    TiXmlElement *enew;
    if(file!="") {
      if(validate(nslocation, file.c_str())!=0) return 1;
      TiXmlDocument *doc=new TiXmlDocument;
      doc->LoadFile(file.c_str());
      enew=doc->FirstChildElement();
      incorporateNamespace(enew, nsprefix);
    }
    else // or take the child element (as a clone, because the embed element is deleted)
      enew=(TiXmlElement*)e->FirstChildElement()->Clone();

    // delete embed element and insert count time the new element
    for(int i=1; i<=count; i++) {
      cout<<"Embed "<<file<<" ("<<i<<"/"<<count<<")"<<endl;
      if(i==1) e=(TiXmlElement*)(e->Parent()->ReplaceChild(e, *enew));
      else e=(TiXmlElement*)(e->Parent()->InsertAfterChild(e, *enew));
      cout<<"Resubstitute "<<counterName<<" with "<<i<<endl;
      ostringstream oss; oss<<i;
      // resubst counterName
      resubst(e, counterName.c_str(), oss.str().c_str());
      // apply embed to new element
      if(embed(e, nsprefix, paramxmlroot)!=0) return 1;
    }
    return 0;
  }

  TiXmlElement *c=e->FirstChildElement();
  while(c) {
    if(embed(c, nsprefix, paramxmlroot)!=0) return 1;
    c=c->NextSiblingElement();
  }
 
  return 0;
}

void convertToSIUnit(TiXmlElement *e, map<string,string> units) {
  if(e->Attribute("unit")) {
    e->FirstChild()->SetValue(replace(units[e->Attribute("unit")], "value", (string("(")+e->GetText()+")").c_str()));
    e->RemoveAttribute("unit");
  }

  TiXmlElement *c=e->FirstChildElement();
  while(c) {
    convertToSIUnit(c, units);
    c=c->NextSiblingElement();
  }
}

void label(TiXmlElement *e) {
  // writeout all text elements
  if(e->GetText())
    e->FirstChild()->SetValue(string("@TEXTB@")+e->GetText()+"@TEXTE@");

  // writeout name and ref* attributes elements
  TiXmlAttribute *a=e->FirstAttribute();
  for(TiXmlAttribute *a=e->FirstAttribute(); a!=0; a=a->Next()) {
    if(a->Name()==string("name") || string(a->Name()).substr(0,3)=="ref")
      a->SetValue("@ATTRB@"+a->ValueStr()+"@ATTRE@");
  }

  TiXmlElement *c=e->FirstChildElement();
  while(c) {
    label(c);
    c=c->NextSiblingElement();
  }
}

int main(int argc, char *argv[]) {
  if((argc-1)%3!=0 || argc<=1) {
    cout<<"Usage:"<<endl
        <<"mbxmlutilspp <param-file> <main-file> <namespace-location-of-main-file>"<<endl
        <<"             [<param-file> <main-file> <namespace-location-of-main-file>] ..."<<endl
        <<"  The output file is named '.pp.<main-file>'."<<endl
        <<"  Use 'none' if not <param-file> is avaliabel."<<endl
        <<""<<endl
        <<"Copyright (C) 2009 Markus Friedrich <mafriedrich@users.berlios.de>"<<endl
        <<"This is free software; see the source for copying conditions. There is NO"<<endl
        <<"warranty; not even for MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE."<<endl
        <<""<<endl
        <<"Licensed under the GNU Lesser General Public License (LGPL)"<<endl;
    return 0;
  }

  // loop over all files
  for(int nr=0; nr<(argc-1)/3; nr++) {
    char *paramxml=argv[3*nr+1];
    char *mainxml=argv[3*nr+2];
    nslocation=argv[3*nr+3];

    // validate parameter file
    if(string(paramxml)!="none")
      if(validate(SCHEMADIR"/parameter.xsd", paramxml)!=0) return 1;

    // validate main file
    if(validate(nslocation, mainxml)!=0) return 1;

    // read parameter file
    TiXmlElement *paramxmlroot=0;
    if(string(paramxml)!="none") {
      cout<<"Read parameter file"<<endl;
      TiXmlDocument *paramxmldoc=new TiXmlDocument;
      paramxmldoc->LoadFile(paramxml);
      paramxmlroot=paramxmldoc->FirstChildElement();
      map<string,string> dummy;
      incorporateNamespace(paramxmlroot,dummy);
    }

    // convert parameter file to octave notation
    cout<<"Convert xml[Matrix|Vector] and ascii[Matrix|Vector]Ref elements in parameter file to octave format"<<endl;
    if(string(paramxml)!="none")
      if(toOctave(paramxmlroot)!=0) return 1;

    // embed/validate files
    TiXmlDocument *mainxmldoc=new TiXmlDocument;
    mainxmldoc->LoadFile(mainxml);
    TiXmlElement *mainxmlroot=mainxmldoc->FirstChildElement();
    map<string,string> nsprefix;
    incorporateNamespace(mainxmlroot,nsprefix);
    if(embed(mainxmlroot,nsprefix,paramxmlroot)!=0) return 1;

    // validate embeded file
    cout<<"Parse and validate embeded file"<<endl;
    unIncorporateNamespace(mainxmlroot, nsprefix);
    mainxmldoc->SaveFile(".mbxmlutilspp.xml");
    if(validate(nslocation, ".mbxmlutilspp.xml")!=0) return 1;
    incorporateNamespace(mainxmlroot, nsprefix);

    // convert main file to octave notation
    cout<<"Convert xml[Matrix|Vector] and ascii[Matrix|Vector]Ref elements to octave format"<<endl;
    if(toOctave(mainxmlroot)!=0) return 1;
    
    // resubst parameters
    if(string(paramxml)!="none") {
      cout<<"Resubstitute parameters"<<endl;
      TiXmlElement *el=paramxmlroot;
      for(el=el->FirstChildElement(); el!=0; el=el->NextSiblingElement())
        resubst(mainxmlroot, el->Attribute("name"), (string("(")+el->GetText()+")").c_str());
    }

    // convert to SI unit
    // get units
    cout<<"Convert all 'physicalvariable' to SI unit"<<endl;
    TiXmlDocument *mmdoc=new TiXmlDocument;
    mmdoc->LoadFile(XMLDIR"/measurement.xml");
    TiXmlElement *ele, *el2;
    map<string,string> units;
    for(ele=mmdoc->FirstChildElement()->FirstChildElement(); ele!=0; ele=ele->NextSiblingElement())
      for(el2=ele->FirstChildElement(); el2!=0; el2=el2->NextSiblingElement()) {
        if(units.find(el2->Attribute("name"))!=units.end()) {
          cout<<"ERROR! Unit name "<<el2->Attribute("name")<<" is defined more than once."<<endl;
          return 1;
        }
        units[el2->Attribute("name")]=el2->GetText();
      }
    // convert units
    convertToSIUnit(mainxmlroot, units);

    // label expressions
    cout<<"Label all 'physicalvariable'"<<endl;
    label(mainxmlroot);

    // save file
    cout<<"Save labeled file"<<endl;
    unIncorporateNamespace(mainxmlroot, nsprefix);
    mainxmldoc->SaveFile(".mbxmlutilspp.xml");

    // call octave file
    cout<<"Process labeled file by octave"<<endl;
    if(system((OCTAVE" -q "OCTAVEDIR"/evaluate.m .mbxmlutilspp.xml .pp."+string(mainxml)).c_str())!=0) return 1;
  }

  return 0;
}
