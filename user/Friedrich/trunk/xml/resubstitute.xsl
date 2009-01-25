<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="2.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:pa="http://www.amm.mw.tu-muenchen.de/XXX/parameter"
  xmlns:pv="http://www.amm.mw.tu-muenchen.de/XXX/physicalvariable">

  <xsl:output method="xml" version="1.0" indent="yes"/>

  <!-- clone -->

  <xsl:template match="*">
    <xsl:copy>
      <xsl:apply-templates select="@*"/>
      <xsl:apply-templates/>
    </xsl:copy>
  </xsl:template>

  <xsl:template match="@*">
    <xsl:copy/>
  </xsl:template>



  <!-- deparam (recursive) -->

  <xsl:import-schema namespace="@NAMESPACE@" schema-location="@NAMESPACELOCATION@"/>

  <xsl:template match="element(*,pv:scalar)|element(*,pv:vector)|element(*,pv:matrix)">
    <xsl:copy>
      <xsl:apply-templates select="@*"/>
      <!-- call deparam template with initial expresion of the text node;
                                 with first [scalar|vector|matrix]Param node;
                                 initialize the changed flag to false -->
      <xsl:call-template name="PARAM">
        <xsl:with-param name="EXPRESSION" select="."/>
        <xsl:with-param name="NODE" select="document('.parameter.octavestring.xml')/pa:parameter/*[1]"/>
        <xsl:with-param name="CHANGED" select="false()"/>
        <xsl:with-param name="RECURSIONLEVEL" select="1"/>
      </xsl:call-template>
    </xsl:copy>
  </xsl:template>

  <xsl:template name="PARAM">
    <xsl:param name="EXPRESSION"/>
    <xsl:param name="NODE"/>
    <xsl:param name="CHANGED"/>
    <xsl:param name="RECURSIONLEVEL"/>
    <!-- output high recursion level as warning -->
    <xsl:if test="$RECURSIONLEVEL mod 50=0">
      <xsl:message>WARNING! Recursion level of parameter substitution is now <xsl:value-of select="$RECURSIONLEVEL"/>. Maybe a infinit loop!</xsl:message>
    </xsl:if>
    <!-- if $NODE exist call deparam template with subst expression and next node;
         set changed flag to true if a subst has occured -->
    <xsl:if test="$NODE">
      <xsl:call-template name="PARAM">
        <xsl:with-param name="EXPRESSION" select="replace($EXPRESSION,concat('([^a-zA-Z0-9_])',$NODE/@name,'([^a-zA-Z0-9_])'),concat('$1(',normalize-space($NODE),')$2'))"/>
        <xsl:with-param name="NODE" select="$NODE/following-sibling::*[1]"/>
        <xsl:with-param name="CHANGED" select="$CHANGED or matches($EXPRESSION,concat('([^a-zA-Z0-9_])',$NODE/@name,'([^a-zA-Z0-9_])'))"/>
        <xsl:with-param name="RECURSIONLEVEL" select="$RECURSIONLEVEL"/>
      </xsl:call-template>
    </xsl:if>
    <!-- if $NODE doen't exist -->
    <xsl:if test="not($NODE)">
      <!-- if a change has occured in last cycle call deparam template with current expession;
           the first param node;
           and set changed state to false
           (start a new recursive deparam cycle) -->
      <xsl:if test="$CHANGED=true()">
        <xsl:call-template name="PARAM">
          <xsl:with-param name="EXPRESSION" select="$EXPRESSION"/>
          <xsl:with-param name="NODE" select="document('.parameter.octavestring.xml')/pa:parameter/*[1]"/>
          <xsl:with-param name="CHANGED" select="false"/>
          <xsl:with-param name="RECURSIONLEVEL" select="$RECURSIONLEVEL+1"/>
        </xsl:call-template>
      </xsl:if>
      <!-- if no change has occured in last cycle output expression end leave -->
      <xsl:if test="$CHANGED=false()">
        <xsl:value-of select="$EXPRESSION"/>
      </xsl:if>
    </xsl:if>
  </xsl:template>

</xsl:stylesheet>
