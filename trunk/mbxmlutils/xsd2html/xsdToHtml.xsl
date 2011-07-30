<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xs="http://www.w3.org/2001/XMLSchema"
  xmlns:pv="http://openmbv.berlios.de/MBXMLUtils/physicalvariable"
  xmlns:html="http://www.w3.org/1999/xhtml"
  xmlns="http://www.w3.org/1999/xhtml"
  version="1.0">

  <!-- If changes in this file are made, then the analog changes must
       be done in the file xstToTex.xsl -->

  <xsl:param name="PROJECT"/>
  <xsl:param name="DATETIME"/>
  <xsl:param name="MBXMLUTILSVERSION"/>



  <!-- output method -->
  <xsl:output method="xml"
    encoding="UTF-8"
    doctype-public="-//W3C//DTD XHTML 1.0 Transitional//EN"
    doctype-system="http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd"/>

  <!-- no default text -->
  <xsl:template match="text()"/>


  <!-- all nodes of all imported schemas and myself -->
  <xsl:param name="ALLNODES" select="document(/xs:schema/xs:import/@schemaLocation)|/"/>



  <xsl:template match="/">
    <!-- html header -->
    <html xml:lang="en" lang="en">
    <head>
      <title><xsl:value-of select="$PROJECT"/> - XML Documentation</title>
      <style type="text/css">
        div.para { margin-bottom:1ex }
        dl,dd { }
        dt { font-weight:bold }
        img.eqn { display:block;margin-bottom:1ex;margin-top:1ex }
        img.inlineeqn { vertical-align:middle }
        img.figure { }
        table.figure { }
        div.figure { margin-bottom:1ex;margin-top:1ex }

        h2,h3 { margin-top:10ex;font-size:14pt }
        ul.content { padding-left:3ex;list-style-type:none }
        span.occurance { font-style:italic }

        *.element { font-family:monospace;font-weight:bold }
        *.type { font-family:monospace }
        *.attribute { font-family:monospace;font-weight:bold;margin-left:2ex }
        *.elementdocu { }
        *.elementdocuall { padding-left:3ex;margin:0;margin-bottom:1ex }
        *.classdocu { }
        *.classdocuall { margin-top:2ex;margin-bottom:2ex }
        ul.elementchoice { list-style-type:none;border-left-style:solid;border-left-color:blue;padding:0.1ex;margin-top:0.25ex;margin-bottom:0.25ex }
        *.elementchoicecolor { color:blue }
        ul.elementsequence { list-style-type:none;border-left-style:solid;border-left-color:red;padding:0.1ex;margin-top:0.25ex;margin-bottom:0.25ex }
        *.elementsequencecolor { color:red }
        ul.elementchild { list-style-type:none;padding-left:4ex }
        ul.elementsofclass { list-style-type:none;padding:0 }

        p.footer { text-align:right;font-size:0.7em }
        img.w3cvalid { border:0;vertical-align:top }

        span.expandcollapsecontent { cursor:nw-resize;color:blue;font-family:monospace;font-weight:bold;font-size:1.25em }
        div.expandcollapseexample { cursor:n-resize;color:blue;font-size:0.75em;font-style:italic;padding-top:2em }
        div.expandcollapsedoxygen { cursor:n-resize;color:blue;font-size:0.75em;font-style:italic }
      </style>
      <script type="text/javascript">
        <![CDATA[
        function expandcollapsecontent(c) {
          var ul=c.parentNode.getElementsByTagName('ul')[0];
          if(ul.style.display=="") {
            ul.style.display="none";
            c.firstChild.data="+ ";
            c.style.cursor="se-resize";
          }
          else {
            ul.style.display="";
            c.firstChild.data="- ";
            c.style.cursor="nw-resize";
          }
        }
        function expandcollapseexample(c) {
          var pre=c.nextSibling;
          if(pre.style.display=="") {
            pre.style.display="none";
            c.firstChild.data="Expand Example...:";
            c.style.cursor="s-resize";
          }
          else {
            pre.style.display="";
            c.firstChild.data="Collapse Example:";
            c.style.cursor="n-resize";
          }
        }
        function collapseexamplesonload() {
          var pre=document.getElementsByTagName("pre");
          for(var i=0; i<pre.length; i++) {
            if(pre[i].getAttribute("class")=="expandcollapsethisexample") {
              var c=pre[i].previousSibling;
              pre[i].style.display="none";
              c.firstChild.data="Expand Example...:";
              c.style.cursor="s-resize";
            }
          }
        }
        function expandcollapsedoxygen(c) {
          var pre=c.nextSibling;
          if(pre.style.display=="") {
            pre.style.display="none";
            c.firstChild.data="Expand Doxygen documentation...:";
            c.style.cursor="s-resize";
          }
          else {
            pre.style.display="";
            c.firstChild.data="Collapse Doxygen documentation:";
            c.style.cursor="n-resize";
          }
        }
        function collapsedoxygenonload() {
          var pre=document.getElementsByTagName("div");
          for(var i=0; i<pre.length; i++) {
            if(pre[i].getAttribute("class")=="expandcollapsethisdoxygen") {
              var c=pre[i].previousSibling;
              pre[i].style.display="none";
              c.firstChild.data="Expand Doxygen documentation...:";
              c.style.cursor="s-resize";
            }
          }
        }
        ]]>
      </script>
    </head>
    <body onload="collapseexamplesonload();collapsedoxygenonload()">
    <h1><xsl:value-of select="$PROJECT"/> - XML Documentation</h1>
    <p>XML-Namespace: <i><xsl:value-of select="/xs:schema/@targetNamespace"/></i></p>
    <h2>Contents</h2>
    <ul class="content">
      <li>1 <a name="introduction-content" href="#introduction">Introduction</a></li>
      <li>2 <a name="nomenclature-content" href="#nomenclature">Nomenclature</a>
        <ul class="content">
          <li>2.1 <a name="aelement-content" href="#aelement">An element</a></li>
          <li>2.2 <a name="achoice-content" href="#achoice">A choice of element</a></li>
          <li>2.3 <a name="asequence-content" href="#asequence">A sequence of elements</a></li>
          <li>2.4 <a name="nested-content" href="#nested">Nested sequences/choices</a></li>
          <li>2.5 <a name="childelements-content" href="#childelements">Child Elements</a></li>
        </ul>
      </li>
      <li>3 <a name="elements-content" href="#elements">Elements</a>
        <ul class="content">
          <xsl:for-each select="/xs:schema/xs:element/@substitutionGroup[not(.=/xs:schema/xs:element/@name) and not(.=preceding::*/@substitutionGroup)]">
            <xsl:sort select="."/>
            <li><span class="expandcollapsecontent" onclick="expandcollapsecontent(this)">- </span><a class="element">
              <xsl:attribute name="href"><xsl:apply-templates mode="GENLINK" select="."/></xsl:attribute>
              &lt;<xsl:value-of select="."/>&gt;</a>
              <ul class="content">
                <xsl:apply-templates mode="CONTENT" select="/xs:schema/xs:element[@substitutionGroup=current()]">
                  <xsl:with-param name="LEVEL" select="0"/>
                  <xsl:with-param name="LEVELNR" select="'3'"/>
                  <xsl:sort select="@name"/>
                </xsl:apply-templates>
              </ul>
            </li>
          </xsl:for-each>
          <xsl:apply-templates mode="CONTENT" select="/xs:schema/xs:element[not(@substitutionGroup)]">
            <xsl:with-param name="LEVEL" select="0"/>
            <xsl:with-param name="LEVELNR" select="'3'"/>
            <xsl:sort select="@name"/>
          </xsl:apply-templates>
        </ul>
      </li>
      <li>4 <a name="simpletypes-content" href="#simpletypes">Simple Types</a>
        <xsl:if test="/xs:schema/xs:simpleType">
          <ul class="content">
            <xsl:for-each select="/xs:schema/xs:simpleType">
              <xsl:sort select="@name"/>
              <li><a class="element" name="{@name}-content" href="#{@name}"><xsl:value-of select="@name"/></a></li>
            </xsl:for-each>
          </ul>
        </xsl:if>
      </li>
    </ul>
    <h2>1 <a name="introduction" href="#introduction-content">Introduction:</a></h2>
    <xsl:apply-templates mode="CLASSANNOTATION" select="/xs:schema/xs:annotation/xs:documentation"/>
    <h2>2 <a name="nomenclature" href="#nomenclature-content">Nomenclature:</a></h2>
    <h3>2.1 <a name="aelement" href="#aelement-content">An element</a></h3>
    <p><span class="element">&lt;ElementName&gt;</span> <span class="occurance">[0-2]</span> (Type: <span class="type">elementType</span>)
    <br/><span class="attribute">attrName1</span> <span class="occurance">[required]</span> (Type: <span class="type">typeOfTheAttribute</span>)
    <br/><span class="attribute">attrName2</span> <span class="occurance">[optional]</span> (Type: <span class="type">typeOfTheAttribute</span>)</p>
    <p class="elementdocuall">
      Documentation of the element.
    </p>
    <p>The upper nomenclature defines a XML element named <span class="element">ElementName</span> with (if given) a minimal occurance of 0 and a maximal occurance of 2. The element is of type <span class="type">elementType</span>.<br/>
    A occurance of <span class="occurance">[optional]</span> means <span class="occurance">[0-1]</span>.<br/>
    The element has two attributes named <span class="attribute">attrName1</span> and <span class="attribute">attrName2</span> of type <span class="type">typeOfTheAttribute</span>. A attribute can be optional or required.</p>
    <h3>2.2 <a name="achoice" href="#achoice-content">A choice of element</a></h3>
    <ul class="elementchoice">
      <li class="elementchoicecolor"><span class="occurance">[1-2]</span></li>
      <li><span class="element">&lt;ElemenetA&gt;</span></li>
      <li><span class="element">&lt;ElemenetB&gt;</span></li>
    </ul>
    <p>The upper nomenclature defines a choice of elements. Only one element of the given ones can be used. The choice has, if given, a minimal occurance of 1 and a maximal maximal occurence of 2.<br/>
    A occurance of <span class="occurance">[optional]</span> means <span class="occurance">[0-1]</span>.</p>
    <h3>2.3 <a name="asequence" href="#asequence-content">A sequence of elements</a></h3>
    <ul class="elementsequence">
      <li class="elementsequencecolor"><span class="occurance">[0-3]</span></li>
      <li><span class="element">&lt;ElemenetA&gt;</span></li>
      <li><span class="element">&lt;ElemenetB&gt;</span></li>
    </ul>
    <p>The upper nomenclature defines a sequence of elements. Each element must be given in that order. The sequence has, if given, a minimal occurance of 0 and a maximal maximal occurence of 3.<br/>
    A occurance of <span class="occurance">[optional]</span> means <span class="occurance">[0-1]</span>.</p>
    <h3>2.4 <a name="nested" href="#nested-content">Nested sequences/choices</a></h3>
    <ul class="elementsequence">
      <li class="elementsequencecolor"><span class="occurance">[1-2]</span></li>
      <li><span class="element">&lt;ElemenetA&gt;</span></li>
      <li>
        <ul class="elementchoice">
          <li class="elementchoicecolor"><span class="occurance">[0-3]</span></li>
          <li><span class="element">&lt;ElemenetC&gt;</span></li>
          <li><span class="element">&lt;ElemenetD&gt;</span></li>
        </ul>
      </li>
      <li><span class="element">&lt;ElemenetB&gt;</span></li>
    </ul>
    <p>Sequences and choices can be nested like above.</p>
    <h3>2.5 <a name="childelements" href="#childelements-content">Child Elements</a></h3>
    <ul class="elementsequence">
      <li class="elementsequencecolor"><span class="occurance">[1-2]</span></li>
      <li><span class="element">&lt;ParantElemenet&gt;</span>
        <ul class="elementchild">
          <li>
            <ul class="elementchoice">
              <li class="elementchoicecolor"><span class="occurance">[0-3]</span></li>
              <li><span class="element">&lt;ChildElemenetA&gt;</span></li>
              <li><span class="element">&lt;ChildElemenetB&gt;</span></li>
            </ul>
          </li>
        </ul>
      </li>
    </ul>
    <p>A indent indicates child elements for a given element.</p>

    <h2>3 <a name="elements" href="#elements-content">Elements</a></h2>
    <xsl:for-each select="/xs:schema/xs:element/@substitutionGroup[not(.=/xs:schema/xs:element/@name) and not(.=preceding::*/@substitutionGroup)]">
      <xsl:sort select="."/>
      <!-- heading -->
      <!-- use h3 for all section headings independent of the LEVEL -->
      <h3 class="element"><a>
        <xsl:attribute name="href"><xsl:apply-templates mode="GENLINK" select="."/></xsl:attribute>
        &lt;<xsl:value-of select="."/>&gt;</a></h3>
      <p>This element is defined by the XML Schema (Project) with the namespace
        <i><xsl:value-of select="../namespace::*[name()=substring-before(current(),':')]"/></i>, which is
        included by this XML Schema (Project). See the documentation of the included XML Schema (Project) for this element.</p>
      <xsl:apply-templates mode="WALKCLASS" select="/xs:schema/xs:element[@substitutionGroup=current()]">
        <xsl:with-param name="LEVEL" select="1"/>
        <xsl:with-param name="LEVELNR" select="'3'"/>
        <xsl:sort select="@name"/>
      </xsl:apply-templates>
    </xsl:for-each>
    <xsl:apply-templates mode="WALKCLASS" select="/xs:schema/xs:element[not(@substitutionGroup)]">
      <xsl:with-param name="LEVEL" select="0"/>
      <xsl:with-param name="LEVELNR" select="'3'"/>
      <xsl:sort select="@name"/>
    </xsl:apply-templates>

    <h2>4 <a name="simpletypes" href="#simpletypes-content">Simple Types</a></h2>
    <xsl:apply-templates mode="SIMPLETYPE" select="/xs:schema/xs:simpleType">
      <xsl:sort select="@name"/>
    </xsl:apply-templates>

    <hr/>
    <p class="footer">
      <a href="http://validator.w3.org/check?uri=referer">
        <img class="w3cvalid" src="http://www.w3.org/Icons/valid-xhtml10-blue" alt="Valid XHTML 1.0 Transitional"/>
      </a>
      <a href="http://jigsaw.w3.org/css-validator/check/referer">
        <img class="w3cvalid" src="http://jigsaw.w3.org/css-validator/images/vcss-blue" alt="Valid CSS!"/>
      </a>
      Generated on <xsl:value-of select="$DATETIME"/> for <xsl:value-of select="$PROJECT"/> by <a href="http://openmbv.berlios.de">MBXMLUtils</a><xsl:text> </xsl:text><xsl:value-of select="$MBXMLUTILSVERSION"/>
    </p>
    </body></html>
  </xsl:template>

  <!-- generate html link form a attribute -->
  <xsl:template mode="GENLINK" match="@*">
    <xsl:param name="V1" select="../namespace::*[name()=substring-before(current(),':')]"/>
    <xsl:param name="V2" select="translate($V1,'.:/','___')"/>
    <xsl:text>../</xsl:text><xsl:value-of select="$V2"/><xsl:text>/index.xhtml#</xsl:text>
    <xsl:if test="not(contains(.,':'))">
      <xsl:value-of select="."/>
    </xsl:if>
    <xsl:if test="contains(.,':')">
      <xsl:value-of select="substring-after(.,':')"/>
    </xsl:if>
  </xsl:template>

  <!-- generate contents -->
  <xsl:template mode="CONTENT" match="/xs:schema/xs:element">
    <xsl:param name="LEVEL"/>
    <xsl:param name="LEVELNR"/>
    <xsl:param name="NAME" select="@name"/>
    <li><span class="expandcollapsecontent" onclick="expandcollapsecontent(this)">- </span>
      <xsl:if test="$LEVEL &lt; 0"><!-- prevent numbers -->
        <xsl:value-of select="$LEVELNR"/>.<xsl:value-of select="position()"/>
      </xsl:if>
      <xsl:text> </xsl:text>
      <a class="element" name="{@name}-content" href="#{@name}">&lt;<xsl:value-of select="@name"/>&gt;</a>
      <xsl:if test="/xs:schema/xs:element[@substitutionGroup=$NAME]">
        <ul class="content">
          <xsl:apply-templates mode="CONTENT" select="/xs:schema/xs:element[@substitutionGroup=$NAME]">
            <xsl:with-param name="LEVEL" select="$LEVEL+1"/>
            <xsl:with-param name="LEVELNR" select="concat($LEVELNR,'.',position())"/>
            <xsl:sort select="@name"/>
          </xsl:apply-templates>
        </ul>
      </xsl:if>
    </li>
  </xsl:template>

  <!-- walk throw all elements -->
  <xsl:template mode="WALKCLASS" match="/xs:schema/xs:element">
    <xsl:param name="LEVEL"/>
    <xsl:param name="LEVELNR"/>
    <xsl:param name="NAME" select="@name"/>
    <xsl:apply-templates mode="CLASS" select=".">
      <xsl:with-param name="LEVEL" select="$LEVEL"/>
      <xsl:with-param name="TITLENR" select="concat($LEVELNR,'.',position())"/>
    </xsl:apply-templates>
    <xsl:apply-templates mode="WALKCLASS" select="/xs:schema/xs:element[@substitutionGroup=$NAME]">
      <xsl:with-param name="LEVEL" select="$LEVEL+1"/>
      <xsl:with-param name="LEVELNR" select="concat($LEVELNR,'.',position())"/>
      <xsl:sort select="@name"/>
    </xsl:apply-templates>
  </xsl:template>

  <!-- class -->
  <xsl:template mode="CLASS" match="/xs:schema/xs:element">
    <xsl:param name="LEVEL"/>
    <xsl:param name="TITLENR"/>
    <xsl:param name="TYPENAME" select="@type"/>
    <xsl:param name="CLASSNAME" select="@name"/>
    <!-- heading -->
    <!-- use h3 for all section headings independent of the LEVEL -->
    <h3 class="element">
      <xsl:if test="$LEVEL &lt; 0"><!-- prevent numbers -->
        <xsl:value-of select="$TITLENR"/>
      </xsl:if>
      <xsl:text> </xsl:text>
      <a name="{@name}" href="#{@name}-content">&lt;<xsl:value-of select="@name"/>&gt;</a>
    </h3>
    <table border="1">
      <!-- abstract -->
      <tr><td>Abstract Element:</td><td>
        <xsl:if test="@abstract='true'">true</xsl:if>
        <xsl:if test="@abstract!='true' or not(@abstract)">false</xsl:if>
      </td></tr>
      <!-- inherits -->
      <tr><td>Inherits:</td><td>
        <xsl:if test="@substitutionGroup">
          <a class="element">
            <xsl:attribute name="href"><xsl:apply-templates mode="GENLINK" select="@substitutionGroup"/></xsl:attribute>
            &lt;<xsl:value-of select="@substitutionGroup"/>&gt;</a>
        </xsl:if>
      </td></tr>
      <!-- inherited by -->
      <tr><td>Inherited by:</td><td>
        <xsl:if test="count(/xs:schema/xs:element[@substitutionGroup=$CLASSNAME])>0">
          <xsl:for-each select="/xs:schema/xs:element[@substitutionGroup=$CLASSNAME]">
            <xsl:sort select="@name"/>
            <a class="element" href="#{@name}">&lt;<xsl:value-of select="@name"/>&gt;</a>, 
          </xsl:for-each>
        </xsl:if>
      </td></tr>
      <!-- used in -->
      <!--<tr><td>Can be used in:</td><td><xsl:apply-templates mode="USEDIN2" select="."/></td></tr>-->
      <!-- class attributes -->
      <tr><td>Attributes:</td><td>
      <xsl:apply-templates mode="CLASSATTRIBUTE" select="/xs:schema/xs:complexType[@name=$TYPENAME]/xs:attribute|/xs:schema/xs:complexType[@name=$TYPENAME]/xs:complexContent/xs:extension/xs:attribute"/>
      </td></tr>
    </table>
    <!-- class documentation -->
    <div class="classdocuall">
      <xsl:apply-templates mode="CLASSANNOTATION" select="xs:annotation/xs:documentation"/>
    </div>
    <!-- child elements -->
    <p>Child Elements:</p>
    <!-- child elements for not base class -->
    <xsl:apply-templates mode="CLASS" select="/xs:schema/xs:complexType[@name=$TYPENAME]/xs:complexContent/xs:extension">
      <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
    </xsl:apply-templates>
    <!-- child elements for base class -->
    <xsl:if test="/xs:schema/xs:complexType[@name=$TYPENAME]/xs:sequence|/xs:schema/xs:complexType[@name=$TYPENAME]/xs:choice">
      <ul class="elementsofclass">
        <xsl:apply-templates mode="SIMPLECONTENT" select="/xs:schema/xs:complexType[@name=$TYPENAME]/xs:sequence|/xs:schema/xs:complexType[@name=$TYPENAME]/xs:choice">
          <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
        </xsl:apply-templates>
      </ul>
    </xsl:if>

    <!-- BEGIN show example xml code -->
    <xsl:if test="not(@abstract) or @abstract='false'">
      <div class="expandcollapseexample" onclick="expandcollapseexample(this)">Collapse Example:</div>
      <pre class="expandcollapsethisexample">
      <xsl:apply-templates mode="EXAMPLEELEMENT" select=".">
        <xsl:with-param name="FULLNAME" select="concat('{',namespace::*[name()=substring-before(current/@name,':')],'}',translate(substring(@name,string-length(substring-before(@name,':'))+1),':',''))"/>
        <!-- this FULLNAME is equal to select="@name" with full namespace awareness -->
        <xsl:with-param name="INDENT" select="''"/>
        <xsl:with-param name="CURRENTNS" select="''"/>
      </xsl:apply-templates><xsl:text>
</xsl:text>
      <xsl:apply-templates mode="EXAMPLECHILDS" select=".">
        <xsl:with-param name="INDENT" select="'  '"/>
        <xsl:with-param name="CURRENTNS" select="namespace::*[name()=substring-before(current()/@name,':')]"/>
      </xsl:apply-templates>
      <xsl:text>&lt;/</xsl:text><xsl:value-of select="translate(substring(@name,string-length(substring-before(@name,':'))+1),':','')"/><xsl:text>&gt;
</xsl:text></pre>
    </xsl:if>
    <!-- END show example xml code -->
  </xsl:template>

  <!-- simple type -->
  <xsl:template mode="SIMPLETYPE" match="/xs:schema/xs:simpleType">
    <h3 class="element">
      <a name="{@name}" href="#{@name}-content"><xsl:value-of select="@name"/></a>
    </h3>
    <!-- simpleType documentation -->
    <xsl:apply-templates mode="CLASSANNOTATION" select="xs:annotation/xs:documentation"/>
  </xsl:template>

  <!-- class attributes -->
  <xsl:template mode="CLASSATTRIBUTE" match="/xs:schema/xs:complexType/xs:attribute|/xs:schema/xs:complexType/xs:complexContent/xs:extension/xs:attribute">
    <xsl:if test="@name">
      <span class="element"><xsl:value-of select="@name"/></span>
      <xsl:if test="@use='required'">
        <span class="occurance"> [required]</span><xsl:text> </xsl:text>
      </xsl:if>
      <xsl:if test="@use!='required'">
        <span class="occurance"> [optional]</span><xsl:text> </xsl:text>
      </xsl:if>
      (Type: <a class="type">
        <xsl:attribute name="href"><xsl:apply-templates mode="GENLINK" select="@type"/></xsl:attribute>
        <xsl:value-of select="@type"/></a>)
      <br/>
    </xsl:if>
    <xsl:if test="@ref">
      <a class="element">
        <xsl:attribute name="href"><xsl:apply-templates mode="GENLINK" select="@ref"/></xsl:attribute>
        <xsl:value-of select="@ref"/></a><br/>
    </xsl:if>
  </xsl:template>

  <!-- used in -->
  <!--<xsl:template mode="USEDIN2" match="/xs:schema/xs:element">
      <xsl:param name="SUBSTGROUP" select="@substitutionGroup"/>
      <xsl:param name="CLASSNAME" select="@name"/>
      <xsl:apply-templates mode="USEDIN" select="/descendant::xs:element[@ref=$CLASSNAME]"/>
      <xsl:apply-templates mode="USEDIN2" select="/xs:schema/xs:element[@name=$SUBSTGROUP]"/>
  </xsl:template>
  <xsl:template mode="USEDIN" match="xs:element">
    <xsl:apply-templates mode="USEDIN" select="ancestor::xs:complexType[last()]"/>
  </xsl:template>
  <xsl:template mode="USEDIN" match="xs:complexType">
    <xsl:param name="CLASSTYPE" select="@name"/>
    <a class="element">
      <xsl:attribute name="href"><xsl:apply-templates mode="GENLINK" select="/xs:schema/xs:element[@type=$CLASSTYPE]/@name"/></xsl:attribute>
      &lt;<xsl:value-of select="/xs:schema/xs:element[@type=$CLASSTYPE]/@name"/>&gt;</a>,
  </xsl:template>-->

  <!-- child elements for not base class -->
  <xsl:template mode="CLASS" match="xs:extension">
    <xsl:param name="CLASSNAME"/>
    <ul class="elementsofclass">
      <!-- elements from base class -->
      <li>
        All Elements from 
        <a class="element">
          <xsl:attribute name="href"><xsl:apply-templates mode="GENLINK" select="/xs:schema/xs:element[@name=$CLASSNAME]/@substitutionGroup"/></xsl:attribute>
          &lt;<xsl:value-of select="/xs:schema/xs:element[@name=$CLASSNAME]/@substitutionGroup"/>&gt;</a>
      </li>
      <!-- elements from this class -->
      <xsl:apply-templates mode="SIMPLECONTENT" select="xs:sequence|xs:choice">
        <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
      </xsl:apply-templates>
    </ul>
  </xsl:template>



  <!-- child elements -->
  <xsl:template mode="SIMPLECONTENT" match="xs:complexType">
    <xsl:if test="xs:sequence|xs:choice">
      <ul class="elementchild">
        <xsl:apply-templates mode="SIMPLECONTENT" select="xs:sequence|xs:choice"/>
      </ul>
    </xsl:if>
  </xsl:template>

  <!-- occurance -->
  <xsl:template mode="OCCURANCE" match="xs:sequence|xs:choice|xs:element">
    <xsl:param name="ELEMENTNAME"/>
    <xsl:param name="COLORSTYLE"/>
    <xsl:if test="@minOccurs|@maxOccurs">
      <xsl:element name="{$ELEMENTNAME}" class="{$COLORSTYLE}">
        <xsl:if test="@minOccurs=0 and not(@maxOccurs)">
          <span class="occurance">[optional]</span>
        </xsl:if>
        <xsl:if test="not(@minOccurs=0 and not(@maxOccurs))">
          <span class="occurance">[<xsl:if test="@minOccurs"><xsl:value-of select="@minOccurs"/></xsl:if><xsl:if test="not(@minOccurs)">1</xsl:if>-<xsl:if test="@maxOccurs"><xsl:value-of select="@maxOccurs"/></xsl:if><xsl:if test="not(@maxOccurs)">1</xsl:if>]</span>
        </xsl:if>
      </xsl:element>
    </xsl:if>
  </xsl:template>

  <!-- sequence -->
  <xsl:template mode="SIMPLECONTENT" match="xs:sequence">
    <xsl:param name="CLASSNAME"/>
    <li>
      <ul class="elementsequence">
        <xsl:apply-templates mode="OCCURANCE" select=".">
          <xsl:with-param name="ELEMENTNAME" select="'li'"/>
          <xsl:with-param name="COLORSTYLE" select="'elementsequencecolor'"/>
        </xsl:apply-templates>
        <xsl:apply-templates mode="SIMPLECONTENT" select="xs:element|xs:sequence|xs:choice">
          <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
        </xsl:apply-templates>
      </ul>
    </li>
  </xsl:template>

  <!-- choice -->
  <xsl:template mode="SIMPLECONTENT" match="xs:choice">
    <xsl:param name="CLASSNAME"/>
    <li>
      <ul class="elementchoice">
        <xsl:apply-templates mode="OCCURANCE" select=".">
          <xsl:with-param name="ELEMENTNAME" select="'li'"/>
          <xsl:with-param name="COLORSTYLE" select="'elementchoicecolor'"/>
        </xsl:apply-templates>
        <xsl:apply-templates mode="SIMPLECONTENT" select="xs:element|xs:sequence|xs:choice">
          <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
        </xsl:apply-templates>
      </ul>
    </li>
  </xsl:template>

  <!-- element -->
  <xsl:template mode="SIMPLECONTENT" match="xs:element">
    <xsl:param name="FUNCTIONNAME" select="@name"/>
    <xsl:param name="CLASSNAME"/>
    <li>
      <!-- name by not(ref) -->
      <xsl:if test="not(@ref)">
        <span class="element">&lt;<xsl:value-of select="@name"/>&gt;</span>
      </xsl:if>
      <!-- name by ref -->
      <xsl:if test="@ref">
        <a class="element">
          <xsl:attribute name="href"><xsl:apply-templates mode="GENLINK" select="@ref"/></xsl:attribute>
          &lt;<xsl:value-of select="@ref"/>&gt;</a>
      </xsl:if><xsl:text> </xsl:text>
      <!-- occurence -->
      <xsl:apply-templates mode="OCCURANCE" select=".">
        <xsl:with-param name="ELEMENTNAME" select="'span'"/>
      </xsl:apply-templates>
      <!-- type -->
      <xsl:if test="@type">(Type: <a class="type">
        <xsl:attribute name="href"><xsl:apply-templates mode="GENLINK" select="@type"/></xsl:attribute>
        <xsl:value-of select="@type"/></a>)</xsl:if>
      <!-- element attributes -->
      <xsl:if test="@name and not(@type)">
        <xsl:apply-templates mode="ELEMENTATTRIBUTE" select="xs:complexType/xs:attribute"/>
      </xsl:if>
      <!-- documentation -->
      <div class="elementdocuall">
        <xsl:apply-templates mode="ELEMENTANNOTATION" select="xs:annotation/xs:documentation"/>
      </div>
      <!-- children -->
      <xsl:if test="@name and not(@type)">
        <xsl:apply-templates mode="SIMPLECONTENT" select="xs:complexType"/>
      </xsl:if>
    </li>
  </xsl:template>

  <!-- element attributes -->
  <xsl:template mode="ELEMENTATTRIBUTE" match="xs:attribute">
    <br/>
    <span class="attribute"><xsl:value-of select="@name"/></span>
    <xsl:if test="@use='required'">
      <span class="occurance"> [required]</span><xsl:text> </xsl:text>
    </xsl:if>
    <xsl:if test="@use!='required'">
      <span class="occurance"> [optional]</span><xsl:text> </xsl:text>
    </xsl:if>
    (Type: <a class="type">
      <xsl:attribute name="href"><xsl:apply-templates mode="GENLINK" select="@type"/></xsl:attribute>
      <xsl:value-of select="@type"/></a>)
  </xsl:template>

  <!-- documentation -->
  <xsl:template mode="CLASSANNOTATION" match="xs:annotation/xs:documentation">
    <!-- add Doxygen documentation dynamically (expand-/colapse-able) if other docucmentation is available -->
    <xsl:if test="@source='doxygen' and count(../xs:documentation[@source!='doxygen' or not(@source)])>0">
      <div class="expandcollapsedoxygen" onclick="expandcollapsedoxygen(this)">Collapse Doxygen docucmentation:</div>
      <div class="expandcollapsethisdoxygen">
        <div class="classdocu"><xsl:apply-templates mode="CLONEDOC"/></div>
      </div>
    </xsl:if>
    <!-- add Doxygen documentation staticlly if no other docucmentation is available -->
    <xsl:if test="@source='doxygen' and count(../xs:documentation[@source!='doxygen' or not(@source)])=0">
      <div class="classdocu"><xsl:apply-templates mode="CLONEDOC"/></div>
    </xsl:if>
    <!-- always add other documentation statically -->
    <xsl:if test="@source!='doxygen' or not(@source)">
      <div class="classdocu"><xsl:apply-templates mode="CLONEDOC"/></div>
    </xsl:if>
  </xsl:template>

  <!-- documentation -->
  <xsl:template mode="ELEMENTANNOTATION" match="xs:annotation/xs:documentation">
    <!-- add Doxygen documentation dynamically (expand-/colapse-able) if other docucmentation is available -->
    <xsl:if test="@source='doxygen' and count(../xs:documentation[@source!='doxygen' or not(@source)])>0">
      <div class="expandcollapsedoxygen" onclick="expandcollapsedoxygen(this)">Collapse Doxygen docucmentation:</div>
      <div class="expandcollapsethisdoxygen">
        <div class="elementdocu">
          <xsl:apply-templates mode="CLONEDOC"/>
        </div>
      </div>
    </xsl:if>
    <!-- add Doxygen documentation staticlly if no other docucmentation is available -->
    <xsl:if test="@source='doxygen' and count(../xs:documentation[@source!='doxygen' or not(@source)])=0">
      <div class="elementdocu">
        <xsl:apply-templates mode="CLONEDOC"/>
      </div>
    </xsl:if>
    <!-- always add other documentation statically -->
    <xsl:if test="@source!='doxygen' or not(@source)">
      <div class="elementdocu">
        <xsl:apply-templates mode="CLONEDOC"/>
      </div>
    </xsl:if>
  </xsl:template>

  <!-- clone doxygen xml/html part -->
  <xsl:template mode="CLONEDOC" match="*">
    <xsl:copy>
      <xsl:for-each select="@*">
        <xsl:copy/>
      </xsl:for-each>
      <xsl:apply-templates mode="CLONEDOC"/>
    </xsl:copy>
  </xsl:template>
  <xsl:template mode="CLONEDOC" match="text()">
    <xsl:copy/>
  </xsl:template>
  <xsl:template mode="CLONEDOC" match="html:object[@class='eqn']">
    <img class="eqn" src="{concat('mbxmlutils_',generate-id(),'.png')}" alt="{.}"/>
  </xsl:template>
  <xsl:template mode="CLONEDOC" match="html:object[@class='inlineeqn']">
    <img class="inlineeqn" src="{concat('mbxmlutils_',generate-id(),'.png')}" alt="{.}"/>
  </xsl:template>
  <xsl:template mode="CLONEDOC" match="html:object[@class='figure']">
    <div class="figure">
      <table class="figure">
        <caption align="bottom"><xsl:value-of select="@title"/></caption>
        <tr><td><img class="figure" src="{@data}.png" alt="{@data}"></img></td></tr>
      </table>
    </div>
  </xsl:template>
  <xsl:template mode="CLONEDOC" match="html:object[@class='figure_html']">
    <div class="figure">
      <table class="figure">
        <caption align="bottom"><xsl:value-of select="@title"/></caption>
        <tr><td><img class="figure" src="{@data}" alt="{@data}"></img></td></tr>
      </table>
    </div>
  </xsl:template>
  <xsl:template mode="CLONEDOC" match="html:object[@class='figure_latex']">
  </xsl:template>
  <xsl:template mode="CLONEDOC" match="html:a[@class='link']">
    <a><xsl:attribute name="href"><xsl:apply-templates mode="GENLINK" select="@href"/></xsl:attribute>
    <xsl:value-of select="."/></a>
  </xsl:template>

  <!-- BEGIN show example xml code -->
  <xsl:template mode="EXAMPLEELEMENT" match="/xs:schema/xs:element">
    <xsl:param name="FULLNAME"/>
    <xsl:param name="INDENT"/>
    <xsl:param name="CURRENTNS"/>
    <xsl:param name="NS_SUBSTITUTIONGROUP" select="namespace::*[name()=substring-before(current()/@substitutionGroup,':')]"/>
    <xsl:param name="NAME_SUBSTITUTIONGROUP" select="translate(substring(@substitutionGroup,string-length(substring-before(@substitutionGroup,':'))+1),':','')"/>
    <xsl:if test="concat('{',namespace::*[name()=substring-before(current/@name,':')],'}',translate(substring(@name,string-length(substring-before(@name,':'))+1),':',''))=$FULLNAME">
      <xsl:value-of select="$INDENT"/>&lt;<xsl:value-of select="translate(substring(@name,string-length(substring-before(@name,':'))+1),':','')"/>
    </xsl:if>
    <xsl:apply-templates mode="EXAMPLEELEMENT" select="$ALLNODES/xs:schema/xs:element[concat('{',namespace::*[name()=substring-before(../@name,':')],'}',translate(substring(@name,string-length(substring-before(@name,':'))+1),':',''))=concat('{',$NS_SUBSTITUTIONGROUP,'}',$NAME_SUBSTITUTIONGROUP)]">
      <!-- this apply-templates is equal to select="/xs:schema/xs:element[@name=current()/@substitutionGroup]" with namespace aware attribute values -->
      <xsl:with-param name="FULLNAME" select="$FULLNAME"/>
      <xsl:with-param name="INDENT" select="$INDENT"/>
      <xsl:with-param name="CURRENTNS" select="$CURRENTNS"/>
    </xsl:apply-templates>
    <xsl:for-each select="/xs:schema/xs:complexType[@name=current()/@type]/xs:complexContent/xs:extension/xs:attribute|/xs:schema/xs:complexType[@name=current()/@type]/xs:attribute">
      <xsl:if test="@name!='xml:base'"> <!-- do not output the (internal) xml:base attribute -->
        <xsl:text> </xsl:text><xsl:value-of select="@name"/><xsl:value-of select="@ref"/>="VALUE"<xsl:text></xsl:text>
      </xsl:if>
    </xsl:for-each>
    <xsl:if test="concat('{',namespace::*[name()=substring-before(current/@name,':')],'}',translate(substring(@name,string-length(substring-before(@name,':'))+1),':',''))=$FULLNAME">
      <xsl:apply-templates mode="XXX" select=".">
        <xsl:with-param name="CURRENTNS" select="$CURRENTNS"/>
      </xsl:apply-templates>&gt;<xsl:text></xsl:text>
    </xsl:if>
  </xsl:template>
  <xsl:template mode="XXX" match="*">
    <xsl:param name="CURRENTNS"/>
    <xsl:if test="@name">
      <xsl:if test="$CURRENTNS!=namespace::*[name()=substring-before(current()/@name,':')]">
        <xsl:text> xmlns="</xsl:text><xsl:value-of select="namespace::*[name()=substring-before(current()/@name,':')]"/>"<xsl:text></xsl:text>
      </xsl:if>
    </xsl:if>
    <xsl:if test="@ref">
      <xsl:if test="$CURRENTNS!=namespace::*[name()=substring-before(current()/@ref,':')]">
        <xsl:text> xmlns="</xsl:text><xsl:value-of select="namespace::*[name()=substring-before(current()/@ref,':')]"/>"<xsl:text></xsl:text>
      </xsl:if>
    </xsl:if>
  </xsl:template>

  <xsl:template mode="EXAMPLECHILDS" match="/xs:schema/xs:element">
    <xsl:param name="INDENT"/>
    <xsl:param name="NS_SUBSTITUTIONGROUP" select="namespace::*[name()=substring-before(current()/@substitutionGroup,':')]"/>
    <xsl:param name="NAME_SUBSTITUTIONGROUP" select="translate(substring(@substitutionGroup,string-length(substring-before(@substitutionGroup,':'))+1),':','')"/>
    <xsl:param name="CURRENTNS"/>
    <xsl:apply-templates mode="EXAMPLECHILDS" select="$ALLNODES/xs:schema/xs:element[concat('{',namespace::*[name()=substring-before(../@name,':')],'}',translate(substring(@name,string-length(substring-before(@name,':'))+1),':',''))=concat('{',$NS_SUBSTITUTIONGROUP,'}',$NAME_SUBSTITUTIONGROUP)]">
      <!-- this apply-templates is equal to select="/xs:schema/xs:element[@name=current()/@substitutionGroup]" with namespace aware attribute values -->
      <xsl:with-param name="INDENT" select="$INDENT"/>
      <xsl:with-param name="CURRENTNS" select="$CURRENTNS"/>
    </xsl:apply-templates>
    <xsl:apply-templates mode="EXAMPLELOCAL" select="/xs:schema/xs:complexType[@name=current()/@type]/xs:sequence|/xs:schema/xs:complexType[@name=current()/@type]/xs:choice|/xs:schema/xs:complexType[@name=current()/@type]/xs:element|/xs:schema/xs:complexType[@name=current()/@type]/xs:complexContent/xs:extension/xs:sequence|/xs:schema/xs:complexType[@name=current()/@type]/xs:complexContent/xs:extension/xs:choice|/xs:schema/xs:complexType[@name=current()/@type]/xs:complexContent/xs:extension/xs:element">
      <xsl:with-param name="INDENT" select="$INDENT"/>
      <xsl:with-param name="CURRENTNS" select="$CURRENTNS"/>
    </xsl:apply-templates>
  </xsl:template>

  <xsl:template mode="EXAMPLELOCAL" match="xs:sequence">
    <xsl:param name="INDENT"/>
    <xsl:param name="CURRENTNS"/>
    <xsl:apply-templates mode="EXAMPLELOCAL" select="xs:sequence|xs:choice|xs:element">
      <xsl:with-param name="INDENT" select="$INDENT"/>
      <xsl:with-param name="CURRENTNS" select="$CURRENTNS"/>
    </xsl:apply-templates>
  </xsl:template>

  <xsl:template mode="EXAMPLELOCAL" match="xs:choice">
    <xsl:param name="INDENT"/>
    <xsl:param name="CURRENTNS"/>
    <xsl:apply-templates mode="EXAMPLELOCAL" select="child::*[position()=1]">
      <xsl:with-param name="INDENT" select="$INDENT"/>
      <xsl:with-param name="CURRENTNS" select="$CURRENTNS"/>
    </xsl:apply-templates>
  </xsl:template>

  <xsl:template mode="EXAMPLELOCAL" match="xs:element">
    <xsl:param name="INDENT"/>
    <xsl:param name="NS_REF" select="namespace::*[name()=substring-before(current()/@ref,':')]"/>
    <xsl:param name="NAME_REF" select="translate(substring(@ref,string-length(substring-before(@ref,':'))+1),':','')"/>
    <xsl:param name="CURRENTNS"/>
    <xsl:if test="@ref and $ALLNODES/xs:schema/xs:element[concat('{',namespace::*[name()=substring-before(../@name,':')],'}',translate(substring(@name,string-length(substring-before(@name,':'))+1),':',''))=concat('{',$NS_REF,'}',$NAME_REF) and (@abstract='false' or not(@abstract))]">
      <!-- this test is equal to test="@ref and /xs:schema/xs:element[@name=current()/@ref and (@abstract='false' or not(@abstract))]" -->
      <xsl:apply-templates mode="EXAMPLEELEMENT" select="$ALLNODES/xs:schema/xs:element[concat('{',namespace::*[name()=substring-before(../@name,':')],'}',translate(substring(@name,string-length(substring-before(@name,':'))+1),':',''))=concat('{',$NS_REF,'}',$NAME_REF)]">
        <!-- this apply-templates is equal to select="/xs:schema/xs:element[@name=current()/@ref]" with namespace aware attribute values -->
        <xsl:with-param name="FULLNAME" select="concat('{',namespace::*[name()=substring-before(current()/@ref,':')],'}',translate(substring(@ref,string-length(substring-before(@ref,':'))+1),':',''))"/>
        <!-- this FULLNAME is equal to select="@ref" with full namespace awareness -->
        <xsl:with-param name="INDENT" select="$INDENT"/>
        <xsl:with-param name="CURRENTNS" select="$CURRENTNS"/>
      </xsl:apply-templates><xsl:apply-templates mode="EXAMPLEOPTIONAL" select="."/><xsl:text>
</xsl:text>
      <xsl:apply-templates mode="EXAMPLECHILDS" select="$ALLNODES/xs:schema/xs:element[concat('{',namespace::*[name()=substring-before(../@name,':')],'}',translate(substring(@name,string-length(substring-before(@name,':'))+1),':',''))=concat('{',$NS_REF,'}',$NAME_REF)]">
        <!-- this apply-templates is equal to select="/xs:schema/xs:element[@name=current()/@ref]" with namespace aware attribute values -->
        <xsl:with-param name="INDENT" select="concat($INDENT,'  ')"/>
        <xsl:with-param name="CURRENTNS" select="namespace::*[name()=substring-before(current()/@ref,':')]"/>
      </xsl:apply-templates>
      <xsl:value-of select="$INDENT"/>&lt;/<xsl:value-of select="translate(substring(@ref,string-length(substring-before(@ref,':'))+1),':','')"/><xsl:text>&gt;
</xsl:text>
    </xsl:if>
    <xsl:if test="not(@ref and $ALLNODES/xs:schema/xs:element[concat('{',namespace::*[name()=substring-before(../@name,':')],'}',translate(substring(@name,string-length(substring-before(@name,':'))+1),':',''))=concat('{',$NS_REF,'}',$NAME_REF) and (@abstract='false' or not(@abstract))])">
    <!-- this test is equal to test="not(@ref and /xs:schema/xs:element[@name=current()/@ref and (@abstract='false' or not(@abstract))])" -->
      <xsl:value-of select="$INDENT"/>&lt;<xsl:value-of select="translate(substring(@name,string-length(substring-before(@name,':'))+1),':','')"/>
      <xsl:value-of select="translate(substring(@ref,string-length(substring-before(@ref,':'))+1),':','')"/>
      <xsl:for-each select="xs:complexType/xs:attribute">
        <xsl:text> </xsl:text><xsl:value-of select="@name"/><xsl:value-of select="@ref"/>="VALUE"<xsl:text></xsl:text>
      </xsl:for-each>
      <xsl:if test="@type">
        <xsl:apply-templates mode="XXX" select=".">
          <xsl:with-param name="CURRENTNS" select="$CURRENTNS"/>
        </xsl:apply-templates>&gt;VALUE&lt;/<xsl:value-of select="translate(substring(@name,string-length(substring-before(@name,':'))+1),':','')"/>&gt;<xsl:apply-templates mode="EXAMPLEOPTIONAL" select="."/><xsl:text>
</xsl:text>
      </xsl:if>
      <xsl:if test="not(@type)">
        <xsl:if test="xs:complexType/xs:sequence|xs:complexType/xs:choice|xs:complexType/xs:element">
          <xsl:apply-templates mode="XXX" select=".">
            <xsl:with-param name="CURRENTNS" select="$CURRENTNS"/>
          </xsl:apply-templates>&gt;<xsl:apply-templates mode="EXAMPLEOPTIONAL" select="."/><xsl:text>
</xsl:text>
          <xsl:apply-templates mode="EXAMPLELOCAL" select="xs:complexType/xs:sequence|xs:complexType/xs:choice|xs:complexType/xs:element">
            <xsl:with-param name="INDENT" select="concat($INDENT, '  ')"/>
            <xsl:with-param name="CURRENTNS" select="$CURRENTNS"/>
          </xsl:apply-templates>
          <xsl:value-of select="$INDENT"/>&lt;/<xsl:value-of select="translate(substring(@name,string-length(substring-before(@name,':'))+1),':','')"/><xsl:value-of select="@ref"/><xsl:text>&gt;
</xsl:text>
        </xsl:if>
        <xsl:if test="not(xs:complexType/xs:sequence|xs:complexType/xs:choice|xs:complexType/xs:element)">
          <xsl:apply-templates mode="XXX" select=".">
            <xsl:with-param name="CURRENTNS" select="$CURRENTNS"/>
          </xsl:apply-templates>/&gt;<xsl:apply-templates mode="EXAMPLEOPTIONAL" select="."/>
          <xsl:if test="$ALLNODES/xs:schema/xs:element[concat('{',namespace::*[name()=substring-before(../@name,':')],'}',translate(substring(@name,string-length(substring-before(@name,':'))+1),':',''))=concat('{',$NS_REF,'}',$NAME_REF) and @abstract='true']"> &lt;!-- Abstract --&gt;</xsl:if>
          <!-- this if is equal test"/xs:schema/xs:element[@name=current()/@ref and @abstract='true']" with full namespace awareness -->
          <xsl:text>
</xsl:text>
        </xsl:if>
      </xsl:if>
    </xsl:if>
  </xsl:template>
  
  <xsl:template mode="EXAMPLEOPTIONAL" match="*">
    <xsl:if test="@minOccurs=0"> &lt;!-- optional --&gt;</xsl:if>
  </xsl:template>
  <!-- END show example xml code -->

</xsl:stylesheet>
