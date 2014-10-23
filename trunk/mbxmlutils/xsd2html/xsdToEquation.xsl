<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  version="1.0">

  <!-- output method -->
  <xsl:output method="text"/>

  <xsl:template match="text()"/>

  <xsl:template match="@*"/>

  <xsl:template match="img[@class='_eqn']"><xsl:text>MBXMLUTILS_BEGINEQN
</xsl:text><xsl:value-of select="@src"/><xsl:text>
\[ </xsl:text><xsl:value-of select="@alt"/><xsl:text> \]
MBXMLUTILS_ENDEQN
</xsl:text></xsl:template>

  <xsl:template match="img[@class='_inlineeqn']"><xsl:text>MBXMLUTILS_BEGINEQN
</xsl:text><xsl:value-of select="@src"/><xsl:text>
$ </xsl:text><xsl:value-of select="@alt"/><xsl:text> $
MBXMLUTILS_ENDEQN
</xsl:text></xsl:template>
 
</xsl:stylesheet>
