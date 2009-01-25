<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="2.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
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



  <!-- dereference pv:xmlVector|pv:xmlMatrix -->

  <xsl:template match="pv:xmlVector|pv:xmlMatrix">
    <xsl:apply-templates mode="XML" select=".."/>
  </xsl:template>

  <xsl:template mode="XML" match="*">
    [ <xsl:apply-templates mode="XML" select="pv:xmlMatrix/pv:row"/>
      <xsl:apply-templates mode="XML" select="pv:xmlVector/pv:ele"/> ]
  </xsl:template>

  <xsl:template mode="XML" match="pv:xmlMatrix/pv:row">
    <xsl:apply-templates mode="XML" select="pv:ele"/>
    <xsl:if test="position()!=last()">; </xsl:if>
  </xsl:template>

  <xsl:template mode="XML" match="pv:xmlMatrix/pv:row/pv:ele">
    <xsl:value-of select="replace(.,' ','')"/> <!-- not not allow space in scalar value -->
    <xsl:if test="position()!=last()">, </xsl:if>
  </xsl:template>

  <xsl:template mode="XML" match="pv:xmlVector/pv:ele">
    <xsl:value-of select="replace(.,' ','')"/> <!-- not not allow space in scalar value -->
    <xsl:if test="position()!=last()">; </xsl:if>
  </xsl:template>



  <!-- dereference pv:xmlMatrixRef|pv:xmlVectorRef -->

  <xsl:template match="pv:xmlVectorRef|pv:xmlMatrixRef">
    <xsl:apply-templates mode="XMLREF" select=".."/>
  </xsl:template>

  <xsl:template mode="XMLREF" match="*">
    [ <xsl:apply-templates mode="XML" select="document(pv:xmlMatrixRef/@href)/pv:xmlMatrix/pv:row"/>
      <xsl:apply-templates mode="XML" select="document(pv:xmlVectorRef/@href)/pv:xmlVector/pv:ele"/> ]
  </xsl:template>



  <!-- dereference pv:asciiMatrixRef|pv:asciiVectorRef -->

  <xsl:template match="pv:asciiScalarRef|pv:asciiVectorRef|pv:asciiMatrixRef">
    <xsl:apply-templates mode="ASCIIREF" select=".."/>
  </xsl:template>

  <xsl:template mode="ASCIIREF" match="*">
    <xsl:variable name="EXP" select="unparsed-text(*/@href)"/> <!-- read ascii file -->
    <xsl:variable name="EXP" select="replace($EXP,'^ *[#%].*$','','m')"/> <!-- delete comments -->
    <xsl:variable name="EXP" select="replace($EXP,'\n','; ')"/> <!-- replace newline with ';' -->
    <xsl:variable name="EXP" select="replace($EXP,';( *;)+',';')"/> <!-- replace sequence of ';' with ';' -->
    <xsl:variable name="EXP" select="replace($EXP,'^ *;','')"/> <!-- delete leading ';' -->
    <xsl:variable name="EXP" select="replace($EXP,'; *$','')"/> <!-- delete trailing ';' -->
    <xsl:if test="not(pv:asciiScalarRef)">[ </xsl:if>
    <xsl:value-of select="$EXP"/>
    <xsl:if test="not(pv:asciiScalarRef)"> ]</xsl:if>
  </xsl:template>

</xsl:stylesheet>
