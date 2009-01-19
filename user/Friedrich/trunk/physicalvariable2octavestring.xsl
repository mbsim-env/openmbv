<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet version="2.0"
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:ty="http://www.amm.mw.tu-muenchen.de/XXX/physicalvariable">

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



  <!-- dereference ty:xmlVector|ty:xmlMatrix -->

  <xsl:template match="ty:xmlVector|ty:xmlMatrix">
    <xsl:apply-templates mode="XML" select=".."/>
  </xsl:template>

  <xsl:template mode="XML" match="*">
    [ <xsl:apply-templates mode="XML" select="ty:xmlMatrix/ty:row"/>
      <xsl:apply-templates mode="XML" select="ty:xmlVector/ty:ele"/> ]
  </xsl:template>

  <xsl:template mode="XML" match="ty:xmlMatrix/ty:row">
    <xsl:apply-templates mode="XML" select="ty:ele"/>
    <xsl:if test="position()!=last()">; </xsl:if>
  </xsl:template>

  <xsl:template mode="XML" match="ty:xmlMatrix/ty:row/ty:ele">
    <xsl:value-of select="replace(.,' ','')"/> <!-- not not allow space in scalar value -->
    <xsl:if test="position()!=last()">, </xsl:if>
  </xsl:template>

  <xsl:template mode="XML" match="ty:xmlVector/ty:ele">
    <xsl:value-of select="replace(.,' ','')"/> <!-- not not allow space in scalar value -->
    <xsl:if test="position()!=last()">; </xsl:if>
  </xsl:template>



  <!-- dereference ty:xmlMatrixRef|ty:xmlVectorRef -->

  <xsl:template match="ty:xmlVectorRef|ty:xmlMatrixRef">
    <xsl:apply-templates mode="XMLREF" select=".."/>
  </xsl:template>

  <xsl:template mode="XMLREF" match="*">
    [ <xsl:apply-templates mode="XML" select="document(ty:xmlMatrixRef/@href)/ty:xmlMatrix/ty:row"/>
      <xsl:apply-templates mode="XML" select="document(ty:xmlVectorRef/@href)/ty:xmlVector/ty:ele"/> ]
  </xsl:template>



  <!-- dereference ty:asciiMatrixRef|ty:asciiVectorRef -->

  <xsl:template match="ty:asciiScalarRef|ty:asciiVectorRef|ty:asciiMatrixRef">
    <xsl:apply-templates mode="ASCIIREF" select=".."/>
  </xsl:template>

  <xsl:template mode="ASCIIREF" match="*">
    <xsl:variable name="EXP" select="unparsed-text(*/@href)"/> <!-- read ascii file -->
    <xsl:variable name="EXP" select="replace($EXP,'^ *[#%].*$','','m')"/> <!-- delete comments -->
    <xsl:variable name="EXP" select="replace($EXP,'\n','; ')"/> <!-- replace newline with ';' -->
    <xsl:variable name="EXP" select="replace($EXP,';( *;)+',';')"/> <!-- replace sequence of ';' with ';' -->
    <xsl:variable name="EXP" select="replace($EXP,'^ *;','')"/> <!-- delete leading ';' -->
    <xsl:variable name="EXP" select="replace($EXP,'; *$','')"/> <!-- delete trailing ';' -->
    <xsl:if test="not(ty:asciiScalarRef)">[ </xsl:if>
    <xsl:value-of select="$EXP"/>
    <xsl:if test="not(ty:asciiScalarRef)"> ]</xsl:if>
  </xsl:template>

</xsl:stylesheet>
