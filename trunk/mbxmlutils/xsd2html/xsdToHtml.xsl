<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:xs="http://www.w3.org/2001/XMLSchema"
  xmlns:pv="http://openmbv.berlios.de/MBXMLUtils/physicalvariable"
  xmlns:html="http://www.w3.org/1999/xhtml"
  xmlns="http://www.w3.org/1999/xhtml"
  version="1.0">

  <xsl:param name="PROJECT"/>
  <xsl:param name="PHYSICALVARIABLEHTMLDOC"/>



  <!-- output method -->
  <xsl:output method="html"
    doctype-public="-//W3C//DTD XHTML 1.0 Transitional//EN"
    doctype-system="http://www.w3.org/TR/xhtml1/DTD/xhtml1-transitional.dtd"/>

  <!-- no default text -->
  <xsl:template match="text()"/>



  <xsl:template match="/">
    <!-- html header -->
    <html xml:lang="en" lang="en">
    <head>
      <title><xsl:value-of select="$PROJECT"/> - XML Documentation</title>
      <style type="text/css">
        div.para { margin-bottom:1ex }
        dl,dd { }
        dt { font-weight:bold }
        div.eqn { margin-bottom:1ex }
        img.eqn { }
        img.inlineeqn { }
        div.htmlfigure,table,caption,tr,td { }
        img.htmlfigure { }
        object.latexfigure { }

        h2,h3,h4,h5,h6,h7,h8,h9 { margin-top:10ex;margin;font-size:14pt }
        ol.content { padding-left:3ex }

        *.element { font-family:monospace;font-weight:bold }
        *.type { font-family:monospace }
        *.attribute { font-family:monospace;font-weight:bold;margin-left:2ex }
        *.elementdocu { padding-left:3ex;margin:0;margin-bottom:1ex }
        *.elementdocu { margin-left:3ex;margin-bottom:1ex }
        *.classdocu { margin-bottom:1ex }
        ul.elementchoice { list-style-type:none;border-left-style:solid;border-left-color:blue;padding:0.1ex;margin-top:0.25ex;margin-bottom:0.25ex }
        *.elementchoicecolor { color:blue }
        ul.elementsequence { list-style-type:none;border-left-style:solid;border-left-color:red;padding:0.1ex;margin-top:0.25ex;margin-bottom:0.25ex }
        *.elementsequencecolor { color:red }
        ul.elementchild { list-style-type:none;padding-left:4ex }
        ul.elementsofclass { list-style-type:none;padding:0 }
      </style>
    </head>
    <body>
    <h1><xsl:value-of select="$PROJECT"/> - XML Documentation</h1>
    <p>This is the Documentation of the XML representation for <xsl:value-of select="$PROJECT"/>.</p>
    <h2>Contents</h2>
    <ol class="content">
      <li><a name="content-nomenclature" href="#nomenclature">Nomenclature</a></li>
      <li>Elements
        <ol class="content">
          <xsl:apply-templates mode="CONTENT" select="/xs:schema/xs:element[not(@substitutionGroup)]">
            <xsl:with-param name="LEVEL" select="0"/>
            <xsl:sort select="@name"/>
          </xsl:apply-templates>
        </ol>
      </li>
    </ol>
    <h2><a name="nomenclature" href="#content-nomenclature">Nomenclature:</a></h2>
    <h3>A element:</h3>
    <p><span class="element">&lt;ElementName&gt;</span> [0-2] (Type: <span class="type">elementType</span>)
    <br/><span class="attribute">attrName1</span><span> [required]</span> (Type: <span class="type">typeOfTheAttribute</span>)
    <br/><span class="attribute">attrName2</span><span> [optional]</span> (Type: <span class="type">typeOfTheAttribute</span>)</p>
    <p class="elementdocu">
      Documentation of the element.
    </p>
    <p>The upper nomenclature defines a XML element named <span class="element">ElementName</span> with (if given) a minimal occurance of 0 and a maximal occurance of 2. The element is of type <span class="type">elementType</span>.<br/>
    A occurance of [optional] means [0-1].<br/>
    The element has two attributes named <span class="attribute">attrName1</span> and <span class="attribute">attrName2</span> of type <span class="type">typeOfTheAttribute</span>. A attribute can be optional or required.</p>
    <h3>A choice of element:</h3>
    <ul class="elementchoice">
      <li class="elementchoicecolor">[1-2]</li>
      <li><span class="element">&lt;ElemenetA&gt;</span></li>
      <li><span class="element">&lt;ElemenetB&gt;</span></li>
    </ul>
    <p>The upper nomenclature defines a choice of elements. Only one element of the given ones can be used. The choice has, if given, a minimal occurance of 1 and a maximal maximal occurence of 2.<br/>
    A occurance of [optional] means [0-1].</p>
    <h3>A seqence of elements:</h3>
    <ul class="elementsequence">
      <li class="elementsequencecolor">[0-3]</li>
      <li><span class="element">&lt;ElemenetA&gt;</span></li>
      <li><span class="element">&lt;ElemenetB&gt;</span></li>
    </ul>
    <p>The upper nomenclature defines a sequence of elements. Each element must be given in that order. The sequence has, if given, a minimal occurance of 0 and a maximal maximal occurence of 3.<br/>
    A occurance of [optional] means [0-1].</p>
    <h3>Nested sequences/choices:</h3>
    <ul class="elementsequence">
      <li class="elementsequencecolor">[1-2]</li>
      <li><span class="element">&lt;ElemenetA&gt;</span></li>
      <li>
        <ul class="elementchoice">
          <li class="elementchoicecolor">[0-3]</li>
          <li><span class="element">&lt;ElemenetC&gt;</span></li>
          <li><span class="element">&lt;ElemenetD&gt;</span></li>
        </ul>
      </li>
      <li><span class="element">&lt;ElemenetB&gt;</span></li>
    </ul>
    <p>Sequences and choices can be nested like above.</p>
    <h3>Child Elements:</h3>
    <ul class="elementsequence">
      <li class="elementsequencecolor">[1-2]</li>
      <li><span class="element">&lt;ParantElemenet&gt;</span>
        <ul class="elementchild">
          <li>
            <ul class="elementchoice">
              <li class="elementchoicecolor">[0-3]</li>
              <li><span class="element">&lt;ChildElemenetA&gt;</span></li>
              <li><span class="element">&lt;ChildElemenetB&gt;</span></li>
            </ul>
          </li>
        </ul>
      </li>
    </ul>
    <p>A indent indicates child elements for a given element.</p>

    <h2>Elements</h2>
    <xsl:apply-templates mode="WALKCLASS" select="/xs:schema/xs:element[not(@substitutionGroup)]">
      <xsl:with-param name="LEVEL" select="0"/>
      <xsl:sort select="@name"/>
    </xsl:apply-templates>
    </body></html>
  </xsl:template>

  <!-- generate contents -->
  <xsl:template mode="CONTENT" match="/xs:schema/xs:element">
    <xsl:param name="LEVEL"/>
    <xsl:param name="NAME" select="@name"/>
    <li>
      <a class="element"><xsl:attribute name="name">content-<xsl:value-of select="@name"/></xsl:attribute>
        <xsl:attribute name="href">#<xsl:value-of select="@name"/></xsl:attribute>&lt;<xsl:value-of select="@name"/>&gt;</a>
      <xsl:if test="/xs:schema/xs:element[@substitutionGroup=$NAME]">
        <ol class="content">
          <xsl:apply-templates mode="CONTENT" select="/xs:schema/xs:element[@substitutionGroup=$NAME]">
            <xsl:with-param name="LEVEL" select="$LEVEL+1"/>
            <xsl:sort select="@name"/>
          </xsl:apply-templates>
        </ol>
      </xsl:if>
    </li>
  </xsl:template>

  <!-- walk throw all elements -->
  <xsl:template mode="WALKCLASS" match="/xs:schema/xs:element">
    <xsl:param name="LEVEL"/>
    <xsl:param name="NAME" select="@name"/>
    <xsl:apply-templates mode="CLASS" select=".">
      <xsl:with-param name="LEVEL" select="$LEVEL"/>
    </xsl:apply-templates>
    <xsl:apply-templates mode="WALKCLASS" select="/xs:schema/xs:element[@substitutionGroup=$NAME]">
      <xsl:with-param name="LEVEL" select="$LEVEL+1"/>
      <xsl:sort select="@name"/>
    </xsl:apply-templates>
  </xsl:template>

  <!-- class -->
  <xsl:template mode="CLASS" match="/xs:schema/xs:element">
    <xsl:param name="LEVEL"/>
    <xsl:param name="TYPENAME" select="@type"/>
    <xsl:param name="CLASSNAME" select="@name"/>
    <!-- heading -->
    <xsl:element name="{concat('h',$LEVEL+3)}">
      <xsl:attribute name="class">element</xsl:attribute>
      <a>
        <xsl:attribute name="name">
          <xsl:value-of select="@name"/>
        </xsl:attribute>
        <xsl:attribute name="href">#content-<xsl:value-of select="@name"/></xsl:attribute>
        &lt;<xsl:value-of select="@name"/>&gt;
      </a>
    </xsl:element>
    <!-- abstract -->
    <xsl:if test="@abstract='true'">
      <p>This element ist abstract.</p>
    </xsl:if>
    <!-- inherits -->
    <xsl:if test="@substitutionGroup">
      <p>
        Inherits:
        <a class="element">
          <xsl:attribute name="href">#<xsl:value-of select="@substitutionGroup"/></xsl:attribute>
          &lt;<xsl:value-of select="@substitutionGroup"/>&gt;</a>
      </p>
    </xsl:if>
    <!-- inherited by -->
    <xsl:if test="count(/xs:schema/xs:element[@substitutionGroup=$CLASSNAME])>0">
      <p>
        Inherited by:
        <xsl:for-each select="/xs:schema/xs:element[@substitutionGroup=$CLASSNAME]">
          <xsl:sort select="@name"/>
          <a class="element">
            <xsl:attribute name="href">#<xsl:value-of select="@name"/></xsl:attribute>
            &lt;<xsl:value-of select="@name"/>&gt;</a><xsl:if test="position()!=last()">, </xsl:if>
        </xsl:for-each>
      </p>
    </xsl:if>
    <!-- used in -->
    <p>Can be used in:
      <xsl:apply-templates mode="USEDIN2" select="."/>
    </p>
    <!-- class documentation -->
    <xsl:apply-templates mode="CLASSANNOTATION" select="xs:annotation/xs:documentation"/>
    <!-- class attributes -->
    <xsl:apply-templates mode="CLASSATTRIBUTE" select="/xs:schema/xs:complexType[@name=$TYPENAME]/xs:attribute"/>
    <!-- child elements -->
    <xsl:if test="/xs:schema/xs:complexType[@name=$TYPENAME]/xs:complexContent/xs:extension/xs:sequence|/xs:schema/xs:complexType[@name=$TYPENAME]/xs:complexContent/xs:extension/xs:choice|/xs:schema/xs:complexType[@name=$TYPENAME]/xs:sequence|/xs:schema/xs:complexType[@name=$TYPENAME]/xs:choice">
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
    </xsl:if>
  </xsl:template>

  <!-- class attributes -->
  <xsl:template mode="CLASSATTRIBUTE" match="/xs:schema/xs:complexType/xs:attribute">
    <p>
      Attribute: <span class="element"><xsl:value-of select="@name"/></span>
      <xsl:if test="@use='required'">
        <span> [required]</span><xsl:text> </xsl:text>
      </xsl:if>
      <xsl:if test="@use!='required'">
        <span> [optional]</span><xsl:text> </xsl:text>
      </xsl:if>
      (Type: <a class="type">
        <xsl:attribute name="href">
          #<xsl:value-of select="@type"/>
        </xsl:attribute>
        <xsl:value-of select="@type"/>
      </a>)
    </p>
  </xsl:template>

  <!-- used in -->
  <xsl:template mode="USEDIN2" match="/xs:schema/xs:element">
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
      <xsl:attribute name="href">#<xsl:value-of select="/xs:schema/xs:element[@type=$CLASSTYPE]/@name"/></xsl:attribute>
      &lt;<xsl:value-of select="/xs:schema/xs:element[@type=$CLASSTYPE]/@name"/>&gt;</a>,
  </xsl:template>

  <!-- child elements for not base class -->
  <xsl:template mode="CLASS" match="xs:extension">
    <xsl:param name="CLASSNAME"/>
    <xsl:if test="xs:sequence|xs:choice">
      <ul class="elementsofclass">
        <!-- elements from base class -->
        <li>
          All Elements from 
          <a class="element">
            <xsl:attribute name="href">#<xsl:value-of select="/xs:schema/xs:element[@name=$CLASSNAME]/@substitutionGroup"/></xsl:attribute>
            &lt;<xsl:value-of select="/xs:schema/xs:element[@name=$CLASSNAME]/@substitutionGroup"/>&gt;</a>
        </li>
        <!-- elements from this class -->
        <xsl:apply-templates mode="SIMPLECONTENT" select="xs:sequence|xs:choice">
          <xsl:with-param name="CLASSNAME" select="$CLASSNAME"/>
        </xsl:apply-templates>
      </ul>
    </xsl:if>
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
      <xsl:element name="{$ELEMENTNAME}">
        <xsl:attribute name="class">
          <xsl:value-of select="$COLORSTYLE"/>
        </xsl:attribute>
        <xsl:if test="@minOccurs=0 and not(@maxOccurs)">
          [optional]
        </xsl:if>
        <xsl:if test="not(@minOccurs=0 and not(@maxOccurs))">
          [<xsl:if test="@minOccurs"><xsl:value-of select="@minOccurs"/></xsl:if><xsl:if test="not(@minOccurs)">1</xsl:if>-<xsl:if test="@maxOccurs"><xsl:value-of select="@maxOccurs"/></xsl:if><xsl:if test="not(@maxOccurs)">1</xsl:if>]
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
          <xsl:attribute name="href">#<xsl:value-of select="@ref"/></xsl:attribute>
          &lt;<xsl:value-of select="@ref"/>&gt;</a>
      </xsl:if><xsl:text> </xsl:text>
      <!-- occurence -->
      <xsl:apply-templates mode="OCCURANCE" select=".">
        <xsl:with-param name="ELEMENTNAME" select="'span'"/>
      </xsl:apply-templates>
      <!-- type -->
      <xsl:if test="@type">
        <!-- type {http://openmbv.berlios.de/MBXMLUTILS/physicalvariable}* -->
        <xsl:if test="substring(@type,1,3)='pv:'">
          (Type: <a class="type">
             <!-- set href to $PHYSICALVARIABLEHTMLDOC#[scalartype|vectortype|matrixtype] -->
            <xsl:attribute name="href"><xsl:value-of select="$PHYSICALVARIABLEHTMLDOC"/>#<xsl:if test="substring(@type,string-length(@type)-5,6)='Scalar'">scalartype</xsl:if>
              <xsl:if test="substring(@type,string-length(@type)-5,6)='Vector'">vectortype</xsl:if>
              <xsl:if test="substring(@type,string-length(@type)-5,6)='Matrix'">matrixtype</xsl:if>
            </xsl:attribute>
            <xsl:value-of select="@type"/>
          </a>)
        </xsl:if>
        <!-- type not {http://openmbv.berlios.de/MBXMLUtils/physicalvariable}* -->
        <xsl:if test="substring(@type,1,3)!='pv:'">
          (Type: <span class="type">
            <xsl:value-of select="@type"/>
          </span>)
        </xsl:if>
      </xsl:if>
      <!-- element attributes -->
      <xsl:if test="@name and not(@type)">
        <xsl:apply-templates mode="ELEMENTATTRIBUTE" select="xs:complexType/xs:attribute"/>
      </xsl:if>
      <!-- documentation -->
      <xsl:apply-templates mode="ELEMENTANNOTATION" select="xs:annotation/xs:documentation"/>
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
      <span> [required]</span><xsl:text> </xsl:text>
    </xsl:if>
    <xsl:if test="@use!='required'">
      <span> [optional]</span><xsl:text> </xsl:text>
    </xsl:if>
    (Type: <span class="type"><xsl:value-of select="@type"/></span>)
  </xsl:template>

  <!-- documentation -->
  <xsl:template mode="CLASSANNOTATION" match="xs:annotation/xs:documentation">
    <xsl:if test="@source='doxygen'">
      <div class="classdocu"><b>The following part is the C++ API docucmentation from Doxygen</b></div>
    </xsl:if>
    <xsl:apply-templates mode="CLONEDOC"/>
  </xsl:template>

  <!-- documentation -->
  <xsl:template mode="ELEMENTANNOTATION" match="xs:annotation/xs:documentation">
    <xsl:if test="@source='doxygen'">
      <div class="elementdocu"><b>The following part is the C++ API docucmentation from Doxygen</b></div>
    </xsl:if>
    <div class="elementdocu">
      <xsl:apply-templates mode="CLONEDOC"/>
    </div>
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

  <xsl:template mode="CLONEDOC" match="html:object[@class='latexfigure']"/>

</xsl:stylesheet>
