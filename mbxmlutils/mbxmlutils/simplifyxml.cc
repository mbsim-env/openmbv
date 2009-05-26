#include <libxml/xmlschemas.h>
#include <iostream>
#include <regex.h>
#include "env.h"
#include "mbxmlutilstinyxml/tinyxml.h"
#include "mbxmlutilstinyxml/tinynamespace.h"

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
  return str;
}
void resubst(TiXmlElement *e, const char *counterName, const char *subst) {
  // resubst all text elements
  if(e->GetText())
    e->FirstChild()->SetValue(replace(e->GetText(), counterName, subst));

  // resubst name and ref* attributes elements
  TiXmlAttribute *a=e->FirstAttribute();
  for(TiXmlAttribute *a=e->FirstAttribute(); a!=0; a=a->Next()) {
    if(a->Name()==string("name"))
      a->SetValue(replace(a->ValueStr(), counterName, subst));
    if(string(a->Name()).substr(0,3)=="ref")
      a->SetValue(replace(a->ValueStr(), counterName, subst));
  }

  TiXmlElement *c=e->FirstChildElement();
  if(c) resubst(c, counterName, subst);
  
  TiXmlElement *s=e->NextSiblingElement();
  if(s) resubst(s, counterName, subst);
}

int embed(TiXmlElement *e, map<string,string> &nsprefix) {
  if(e->ValueStr()==MBXMLUTILSPVNS"embed") {
    string file=e->Attribute("href");
    int count=atoi(e->Attribute("count"));
    string counterName=e->Attribute("counterName");
    if(validate(nslocation, file.c_str())!=0) return 1;
    TiXmlDocument *doc=new TiXmlDocument;
    doc->LoadFile(file.c_str());
    TiXmlElement *enew=doc->FirstChildElement();
    incorporateNamespace(enew, nsprefix);
    TiXmlElement *ee;
    for(int i=1; i<=count; i++) {
      cout<<"Embed "<<file<<" ("<<i<<"/"<<count<<")"<<endl;
      if(i==1) ee=e=(TiXmlElement*)(e->Parent()->ReplaceChild(e, *enew));
      else ee=(TiXmlElement*)(e->Parent()->InsertAfterChild(ee, *enew));
      cout<<"Resubstitute "<<counterName<<" with "<<i<<endl;
      ostringstream oss; oss<<i;
      resubst(ee, counterName.c_str(), oss.str().c_str());
    }
  }

  TiXmlElement *c=e->FirstChildElement();
  if(c)
    if(embed(c, nsprefix)!=0) return 1;
  
  TiXmlElement *s=e->NextSiblingElement();
  if(s)
    if(embed(s, nsprefix)!=0) return 1;
}

void convertToSIUnit(TiXmlElement *e, map<string,string> units) {
  if(e->Attribute("unit")) {
    e->FirstChild()->SetValue(replace(units[e->Attribute("unit")], "value", (string("(")+e->GetText()+")").c_str()));
    e->RemoveAttribute("unit");
  }

  TiXmlElement *c=e->FirstChildElement();
  if(c) convertToSIUnit(c, units);
  
  TiXmlElement *s=e->NextSiblingElement();
  if(s) convertToSIUnit(s, units);
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
  if(c) label(c);
  
  TiXmlElement *s=e->NextSiblingElement();
  if(s) label(s);
}

int main(int argc, char *argv[]) {
  if(argc!=4) {
    cout<<"Usage: mbxmlutils-simplifyxml <param-file> <main-file>"<<endl
        <<"                              <namespace-location-of-main-file>"<<endl
        <<"  The output file is named '.simplified.<main-file>'."<<endl
        <<"  Use 'none' if not <param-file> is avaliabel."<<endl;
    return 0;
  }
  int ret;
  char *paramxml=argv[1];
  char *mainxml=argv[2];
  nslocation=argv[3];

  // validate parameter file
  if(string(paramxml)!="none")
    if(validate(SCHEMADIR"/parameter.xsd", paramxml)!=0) return 1;

  // validate main file
  if(validate(nslocation, mainxml)!=0) return 1;

  // embed/validate files
  TiXmlDocument *mainxmldoc=new TiXmlDocument;
  mainxmldoc->LoadFile(mainxml);
  TiXmlElement *mainxmlroot=mainxmldoc->FirstChildElement();
  map<string,string> nsprefix;
  incorporateNamespace(mainxmlroot,nsprefix);
  if(embed(mainxmlroot,nsprefix)!=0) return 1;

  // validate embeded file
  cout<<"Parse and validate embeded file"<<endl;
  unIncorporateNamespace(mainxmlroot, nsprefix);
  mainxmldoc->SaveFile(".mbxmlutils_simplifyxml.xml");
  if(validate(nslocation, ".mbxmlutils_simplifyxml.xml")!=0) return 1;
  incorporateNamespace(mainxmlroot, nsprefix);

  // convert parameter file to octave notation

  // convert main file to octave notation
  
  // resubst parameters
  if(string(paramxml)!="none") {
    cout<<"Resubstitute parameters"<<endl;
    TiXmlDocument *paramxmldoc=new TiXmlDocument;
    paramxmldoc->LoadFile(paramxml);
    TiXmlElement *el=paramxmldoc->FirstChildElement();
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
  mainxmldoc->SaveFile(".mbxmlutils_simplifyxml.xml");

  // call octave file
  cout<<"Process labeled file by octave"<<endl;
  if(system((OCTAVE" -q "OCTAVEDIR"/evaluate.m .mbxmlutils_simplifyxml.xml .simplified."+string(mainxml)).c_str())!=0) return 1;

  return 0;
}
