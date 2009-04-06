#include "tinynamespace.h"
#include "tinystr.h"

using namespace std;

void incorporateNamespace(TiXmlElement* e, map<string,string> prefixns) {
  // overwrite existing namespace prefixes with new ones
  TiXmlAttribute* a=e->FirstAttribute();
  while(a!=NULL) {
    TiXmlAttribute* aNext=a->Next();
    if(strncmp(a->Name(),"xmlns:",6)==0) {
      // none default prefix
      prefixns[a->Name()+6]=a->ValueStr();
      e->RemoveAttribute(a->Name());
    }
    else if(strcmp(a->Name(),"xmlns")==0) {
      // default prefix
      prefixns[""]=a->ValueStr();
      e->RemoveAttribute(a->Name());
    }
    a=aNext;
  }

  // set element name to '{<namespace>}<localname>'
  if(e->ValueStr().find(':')>=0) {
    for(map<string,string>::iterator i=prefixns.begin(); i!=prefixns.end(); i++) {
      // none default prefix
      if(e->ValueStr().compare(0,(*i).first.length()+1,(*i).first+":")==0)
        e->SetValue("{"+(*i).second+"}"+e->ValueStr().substr((*i).first.length()+1));
      // default prefix
      if(e->ValueStr().find(":")==string::npos)
        e->SetValue("{"+(*i).second+"}"+e->ValueStr());
    }
  }

  if(e->ValueStr()=="{http://www.w3.org/2001/XInclude}include") {
    // for a xi:include element include the href file in the tree
    TiXmlDocument docInclude;
    docInclude.LoadFile(e->Attribute("href"));
    incorporateNamespace(docInclude.FirstChildElement());
    e->Parent()->InsertAfterChild(e,*docInclude.FirstChildElement());
    e->Parent()->RemoveChild(e);
  }
  else {
    // apply recusively for all child elements
    TiXmlElement* c=e->FirstChildElement();
    while(c!=0) {
      TiXmlElement* cNext=c->NextSiblingElement();
      incorporateNamespace(c, prefixns);
      c=cNext;
    }
  }
}

int unIncorporateNamespace(TiXmlElement *e, map<string,string>& nsprefix, bool firstCall) {
  // add namespace prefix attribute to the root element: 'xmlns:...=...'
  if(firstCall)
    for(map<string,string>::iterator i=nsprefix.begin(); i!=nsprefix.end(); i++) {
      if((*i).second!="")
        e->SetAttribute("xmlns:"+(*i).second, (*i).first);
      else
        e->SetAttribute("xmlns", (*i).first);
    }

  // extract namespace form element name
  string ns=e->ValueStr().substr(1,e->ValueStr().find('}')-1);

  // error if the namespace of the element is not found in the map
  if(nsprefix.find(ns)==nsprefix.end())
    return 1;

  // set element name to '<nsprefix>:<localname>'
  e->SetValue(nsprefix[ns]+":"+e->ValueStr().substr(ns.length()+2));
  // delete the leading ':' in case of the default namespace prefix
  if(e->ValueStr().substr(0,1)==":")
    e->SetValue(e->ValueStr().substr(1));

  // apply recusively for all child elements
  int ret=0;
  TiXmlElement* c=e->FirstChildElement();
  while(c!=0) {
    ret+=unIncorporateNamespace(c, nsprefix, false);
    c=c->NextSiblingElement();
  }

  return ret;
}
