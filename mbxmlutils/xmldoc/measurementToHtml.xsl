<xsl:stylesheet
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
  xmlns:mm="http://www.mbsim-env.de/MBXMLUtils/measurement"
  version="1.0">

  <xsl:param name="DATETIME"/>

  <!-- output method -->
  <xsl:output method="html" encoding="UTF-8"/>

  <!-- no default text -->
  <xsl:template match="text()"/>



  <!-- for all direct root child elements (classes) -->
  <xsl:template match="/">
    <!-- html header -->
    <xsl:text disable-output-escaping='yes'>&lt;!DOCTYPE html>
</xsl:text>
    <html lang="en">
    <head>
      <title>MBXMLUtils - XML Documentation</title>
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.6/css/bootstrap.min.css"/>
      <link rel="shortcut icon" href="/mbsim/html/mbsimenv.ico" type="image/x-icon"/>
      <link rel="icon" href="/mbsim/html/mbsimenv.ico" type="image/x-icon"/>
      <!-- Note: all defined class names and function names here start with _ to differentiate them from bootstrap ones -->
      <style>
        ul._content { padding-left:3ex; list-style-type:none; }
        *._attributeNoMargin { font-family:monospace; font-weight:bold; }
        *._type { font-family:monospace; }
        *._element { font-family:monospace; font-weight:bold; }
      </style>

      <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
          extensions: ["tex2jax.js", "TeX/AMSmath.js", "TeX/AMSsymbols.js"],
          jax: ["input/TeX","output/HTML-CSS"],
          tex2jax: {
            inlineMath: [['\\(','\\)']],
            displayMath: [['\\[','\\]']],
          },
        });
      </script>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js"> </script>
    </head>
    <body style="margin:0.5em">
    <script src="https://code.jquery.com/jquery-2.1.4.min.js"> </script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/moment.js/2.22.2/moment.min.js"> </script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/moment-timezone/0.5.23/moment-timezone-with-data-2012-2022.min.js"> </script>
    <script src="/mbsim/html/cookiewarning.js"> </script>
    <script>
      $(document).ready(function() {
        $('.DATETIME').each(function() {
          $(this).text(moment($(this).text()).tz(moment.tz.guess()).format("ddd, YYYY-MM-DD - HH:mm:ss z"));
        }); 
      });
    </script>
    <div class="page-header">
      <h1>MBXMLUtils - XML Documentation</h1>
      <p>XML-Namespace: <span class="label label-warning">http://www.mbsim-env.de/MBXMLUtils</span></p>
    </div>
    <div class="h2">Contents</div>
    <ul class="_content">
      <li><a id="introduction-content" href="#introduction">1 Introduction</a></li>
      <li><a id="legend-content" href="#legend">2 Legend</a></li>
      <li><a id="name-content" href="#name">3 Element Name</a></li>
      <li><a id="type-content" href="#type">4 Types</a>
        <ul class="_content">
          <li><a id="scalartype-content" href="#scalartype">4.1 Scalar Type</a></li>
          <li><a id="vectortype-content" href="#vectortype">4.2 Vector Type</a></li>
          <li><a id="matrixtype-content" href="#matrixtype">4.3 Matrix Type</a></li>
        </ul>
      </li>
      <li><a id="parameters-content" href="#parameters">5 Parameters</a></li>
      <li><a id="evaluator-content" href="#evaluator">6 Expression Evaluator</a></li>
      <li><a id="embed-content" href="#embed">7 Embeding</a></li>
      <li><a id="measurements-content" href="#measurements">8 Measurements</a>
        <ul class="_content">
          <xsl:for-each select="/mm:measurement/mm:measure">
            <xsl:sort select="@name"/>
            <li>
              <a><xsl:attribute name="id"><xsl:value-of select="@name"/>-content</xsl:attribute>
                <xsl:attribute name="href">#<xsl:value-of select="@name"/></xsl:attribute>
                <span class="glyphicon glyphicon-unchecked"/><xsl:text> </xsl:text><xsl:value-of select="@name"/></a>
            </li>
          </xsl:for-each>
        </ul>
      </li>
    </ul>
    <hr class="_hr"/>

    <h1><a id="introduction" href="#introduction-content">1 Introduction</a></h1>

    <h1><a id="legend" href="#legend-content">2 Legend</a></h1>
    <table class="table table-condensed">
      <thead>
        <tr><th>Icon</th><th>Description</th></tr>
      </thead>
      <tbody>
        <tr><td><span class="_element">&lt;element&gt;</span></td><td>A XML element of name 'element'</td></tr>
        <tr><td><span class="_attributeNoMargin">attrName</span></td><td>A XML attribute of name 'attrName'</td></tr>
        <tr><td><span class="label label-warning">namespace</span></td><td>A XML namespace of name 'namespace'</td></tr>
        <tr><td><span class="label label-info">type</span></td><td>A XML element or attribute type of name 'type'</td></tr>
      </tbody>
    </table>

    <h1><a id="name" href="#name-content">3 Element Name</a></h1>
    <p>Elements which must be referable must have a name. Mostly this name is given by the attribute <span class="_attributeNoMargin">name</span>. A valid name starts with a letter or a underscore. The following characters can be letters, underscores or digits. The content between '<code>{</code>' and '<code>}</code>' can be any <a href="#evaluator">Expression Evaluator</a> and is substituted by the result of the evaluator (which must be a valid name; normal a integer number).</p>
    <p>The following table shows examples for valid element names (on the left) and the substituted names (on the right), if there exist a scalar (integer) parameter of name <code>n</code> with the value <code>2</code>:</p>
    <table class="table table-condensed table-striped table-hover">
      <thead>
        <tr><th>Element Name</th><th>Substituted Element Name</th></tr>
      </thead>
      <tbody>
        <tr><td><code>Object1</code></td><td><code>Object1</code></td></tr>
        <tr><td><code>myname_3</code></td><td><code>myname_3</code></td></tr>
        <tr><td><code>body{n+6}</code></td><td><code>body8</code></td></tr>
        <tr><td><code>Obj_{n+4}_{if n==2; ret=3; else ret=6; end}</code></td><td><code>Obj_6_3</code></td></tr>
      </tbody>
    </table>

    <h1><a id="type" href="#type-content">4 Types</a></h1>
    <h2><a id="scalartype" href="#scalartype-content">4.1 Scalar Type</a>
      <xsl:for-each select="/mm:measurement/mm:measure">
        <a id="{@name}Scalar"/>
      </xsl:for-each>
    </h2>
    <p>A scalar type can be of any unit defined in <a href="#measurements">measurements</a>. The unit is given by a optional
      attribute of name <span class="_attributeNoMargin">unit</span>.
      The type name of a scalar of measure length is <span class="label label-info _type">pv:lengthScalar</span> and so on.
      Where <code>pv</code> is mapped to the namespace-uri <span class="label label-warning">http://www.mbsim-env.de/MBXMLUtils</span>.</p>
    <p>The content of a scalar type must be a <a href="#evaluator">Expression Evaluator</a>. The following examples are valid, if there exist a scalar paremter <code>a</code> and <code>b</code> in the <a href="#parameters">parameter file</a>:</p>
    <pre>&lt;myScalarElement unit="mm"&gt;4*sin(a)+b&lt;/myScalarElement&gt;</pre>
    <pre>&lt;myScalarElement&gt;[a,2]*[3;b]&lt;/myScalarElement&gt;</pre>
    <p>There is also a special unit of name <code>unknown</code> defined. This unit dose not take the optional <code>unit</code>
      attribute, it takes an optional attribute of name <span class="_attributeNoMargin">convertUnit</span>. The value of this attribute can be a
      <a href="#evaluator">Expression Evaluator</a>
      which must contain a parameter of name <code>value</code>. The given value is then converted by this expression.</p>

    <h2><a id="vectortype" href="#vectortype-content">4.2 Vector Type</a>
      <xsl:for-each select="/mm:measurement/mm:measure">
        <a id="{@name}Vector"/>
      </xsl:for-each>
    </h2>
    <p>A vector type can be of any unit defined in <a href="#measurements">measurements</a>. The unit is given by a optional
      attribute of name <span class="_attributeNoMargin">unit</span>.
      The type name of a vector of measure length is <span class="label label-info _type">pv:lengthVector</span> and so on.
      Where <code>pv</code> is mapped to the namespace-uri <span class="label label-warning">http://www.mbsim-env.de/MBXMLUtils</span>.</p>
    <p>The content of a vector type can be one of the following:</p>
    <ul>
      <li>A <a href="#evaluator">Expression Evaluator</a>. The following examples are valid, if there exist a scalar paremter <code>a</code> and <code>b</code> in the <a href="#parameters">parameter file</a>:
          <pre>&lt;myVectorElement unit="mm"&gt;[1;b;a;7]&lt;/myVectorElement&gt;</pre>
          <pre>&lt;myVectorElement&gt;[a,2;5.6,7]*[3;b]&lt;/myVectorElement&gt;</pre>
       <p>Using the corresponding evaluator command it is also possible to load the data from a external file (for octave):</p>
       <pre>&lt;myMatrixElement&gt;ret=load('mydata.dat')&lt;/myMatrixElement&gt;</pre>
      </li>
      <li>A XML representation of a vector: The following shows a example of such a XML representation.<pre>
&lt;myVectorElement xmlns:pv="http://www.mbsim-env.de/MBXMLUtils"&gt;
  &lt;pv:ele&gt;6.5&lt;/pv:ele&gt;
  &lt;pv:ele&gt;1.5&lt;/pv:ele&gt;
  &lt;pv:ele&gt;7.3&lt;/pv:ele&gt;
&lt;/myVectorElement&gt;
</pre></li>
    </ul>
    <p>For the special unit of name <code>unknown</code> see <a href="#scalartype">Scalar Type</a></p>

    <h2><a id="matrixtype" href="#matrixtype-content">4.3 Matrix Type</a>
      <xsl:for-each select="/mm:measurement/mm:measure">
        <a id="{@name}Matrix"/>
      </xsl:for-each>
    </h2>
    <p>A matrix type can be of any unit defined in <a href="#measurements">measurements</a>. The unit is given by a optional
      attribute of name <span class="_attributeNoMargin">unit</span>.
      The type name of a matrix of measure length is <span class="label label-info _type">pv:lengthMatrix</span> and so on.
      Where <code>pv</code> is mapped to the namespace-uri <span class="label label-warning">http://www.mbsim-env.de/MBXMLUtils</span>.</p>
    <p>The content of a matrix type can be one of the following:</p>
    <ul>
      <li>A <a href="#evaluator">Expression Evaluator</a>. The following examples are valid, if there exist a scalar paremter <code>a</code> and <code>b</code> in the <a href="#parameters">parameter file</a>:
          <pre>&lt;myMatrixElement unit="mm"&gt;[1,b;a,7]&lt;/myMatrixElement&gt;</pre>
          <pre>&lt;myMatrixElement&gt;[a,2;5.6,7]*rand(2,2)&lt;/myMatrixElement&gt;</pre>
       <p>Using the corresponding evaluator command it is also possible to load the data from a external file (for octave):</p>
       <pre>&lt;myMatrixElement&gt;ret=load('mydata.dat')&lt;/myMatrixElement&gt;</pre>
      </li>
      <li>A XML representation of a matrix: The following shows a example of such a XML representation.<pre>
&lt;myMatrixElement xmlns:pv="http://www.mbsim-env.de/MBXMLUtils"&gt;
  &lt;pv:row&gt;
    &lt;pv:ele&gt;6.5&lt;/pv:ele&gt;
    &lt;pv:ele&gt;1.5&lt;/pv:ele&gt;
  &lt;/pv:row&gt;
  &lt;pv:row&gt;
    &lt;pv:ele&gt;6.5&lt;/pv:ele&gt;
    &lt;pv:ele&gt;1.5&lt;/pv:ele&gt;
  &lt;/pv:row&gt;
&lt;/myMatrixElement&gt;
</pre></li>
    </ul>
    <p>For the special unit of name <code>unknown</code> see <a href="#scalartype">Scalar Type</a></p>

    <h1><a id="parameters" href="#parameters-content">5 Parameters</a></h1>
    <p>A example for a parameter file is given below:</p>
<pre>&lt;Parameter xmlns="http://www.mbsim-env.de/MBXMLUtils"&gt;
  &lt;scalarParameter name="N"&gt;9&lt;/scalarParameter&gt;
  &lt;vectorParameter name="a"&gt;[1;2;3]*N&lt;/scalarParameter&gt;
  &lt;scalarParameter name="lO"&gt;0.2*N&lt;/scalarParameter&gt;
  &lt;matrixParameter name="A"&gt;[1,2;3,4]&lt;/scalarParameter&gt;
&lt;/Parameter&gt;
</pre>
    <p>The parameter names must be unique. The parameters are added from top to bottom. Parameters may depend on parameters already added. The parameter values can be given as <a href="#evaluator">Expression Evaluator</a>. Hence a parameter below another parameter may reference this value.</p>

    <h1><a id="evaluator" href="#evaluator-content">6 Expression Evaluator</a></h1>
    <p>Different expression evaluators can be used. Currently implemented is only octave as evaluator. Hence this section covers only the octave expression evaluator.</p>
    <p>A octave expression/program can be arbitary octave code. So it can be a single statement or a statement list.</p>

   <p>If it is a single statement, then the value for the XML element is just the value of the evaluated octave statement. The type of this value must match the type of the XML element (scalar, vector or matrix). The following examples shows valid examples for a single octave statement (one per line), if a scalar <a href="#parameters">parameter</a> of name <code>a</code> and <code>b</code> exist:</p>
<pre>4
b
3+a*8
[4;a]*[6,b]
</pre>

<p>If the text is a statement list, then the value for the XML element is the value of the variable 'ret' which must be set by the statement list. The type of the variable 'ret' must match the type of the XML element (scalar, vector or matrix). The following examples shows valid examples for a octave statement list (one per line), if a scalar <a href="#parameters">parameter</a> of name <code>a</code> and <code>b</code> exist:</p>
<pre>if 1==a; ret=4; else ret=8; end
myvar=[1;a];myvar2=myvar*2;ret=myvar2*b;dummy=3
</pre>

<p>A octave expression can also expand over more then one line like below. Note that the characters '&amp;', '&lt;' and '&gt;' are not allowed in XML. So you have to quote them with '&amp;amp;', '&amp;lt;' and '&amp;gt;', or you must enclose the octave code in a CDATA section:</p>
<pre>&lt;![CDATA[
function r=myfunc(a)
  r=2*a;
end
if 1 &amp; 2, x=9; else x=8; end
ret=myfunc(m1/2);
]]&gt;
</pre>

<p>The following m-functions extent the octave functionality being useful for several applications:</p>
<dl class="dl-horizontal">
  <dt>\(T=\text{rotateAboutX}(\varphi)\)</dt>
  <dd>Returns the transformation matrix by angle \(\varphi\); around the x-axis:
    \[
      \left[\begin{array}{ccc}
        1 &amp; 0             &amp; 0              \\
        0 &amp; \cos(\varphi) &amp; -\sin(\varphi) \\
        0 &amp; \sin(\varphi) &amp; \cos(\varphi)
      \end{array}\right]
    \]
  </dd>
  <dt>\(T=\text{rotateAboutY}(\varphi)\)</dt>
  <dd>Returns the transformation matrix by angle \(\varphi\); around the y-axis:
    \[
      \left[\begin{array}{ccc}
        \cos(\varphi)  &amp; 0 &amp; \sin(\varphi) \\
        0              &amp; 1 &amp; 0             \\
        -\sin(\varphi) &amp; 0 &amp; \cos(\varphi)
      \end{array}\right]
    \]
  </dd>
  <dt>\(T=\text{rotateAboutZ}(\varphi)\)</dt>
  <dd>Returns the transformation matrix by angle \(\varphi\); around the z-axis:
    \[
      \left[\begin{array}{ccc}
        \cos(\varphi) &amp; -\sin(\varphi) &amp; 0 \\
        \sin(\varphi) &amp; \cos(\varphi)  &amp; 0 \\
        0             &amp; 0              &amp; 1
      \end{array}\right]
    \]
  </dd>
  <dt>\(T=\text{cardan}(\alpha,\beta,\gamma)\)</dt>
  <dd>Returns the cardan transformation matrix:
    \[
      \left[\begin{array}{ccc}
        \cos(\beta)\cos(\gamma)                                      &amp; -\cos(\beta)\sin(\gamma)                                     &amp; \sin(\beta)              \\
        \cos(\alpha)\sin(\gamma)+\sin(\alpha)\sin(\beta)\cos(\gamma) &amp; \cos(\alpha)\cos(\gamma)-\sin(\alpha)\sin(\beta)\sin(\gamma) &amp; -\sin(\alpha)\cos(\beta) \\
        \sin(\alpha)\sin(\gamma)-\cos(\alpha)\sin(\beta)\cos(\gamma) &amp; \cos(\alpha)\sin(\beta)\sin(\gamma)+\sin(\alpha)\cos(\gamma) &amp; \cos(\alpha)\cos(\beta)
      \end{array}\right]
    \]
  </dd>
  <dt>\(T=\text{euler}(\Phi,\theta,\varphi)\)</dt>
  <dd>Returns the Euler transformation matrix:
    \[
      \left[\begin{array}{ccc}
        \cos(\varphi)\cos(\Phi)-\sin(\varphi)\cos(\theta)\sin(\Phi) &amp; -\cos(\varphi)\cos(\theta)\sin(\Phi)-\sin(\varphi)\cos(\Phi) &amp; \sin(\theta)\sin(\Phi)  \\
        \cos(\varphi)\sin(\Phi)+\sin(\varphi)\cos(\theta)\cos(\Phi) &amp; \cos(\varphi)\cos(\theta)\cos(\Phi)-\sin(\varphi)\sin(\Phi)  &amp; -\sin(\theta)\cos(\Phi) \\
        \sin(\varphi)\sin(\theta)                                   &amp; \cos(\varphi)\sin(\theta)                                    &amp; \cos(\theta)
      \end{array}\right]
    \]
  </dd>
</dl>

    <h1><a id="embed" href="#embed-content">7 Embeding</a></h1>
    <p>Using the <span class="_element">&lt;pv:Embed&gt;</span> element, where the prefix <code>pv</code> is mapped to the namespace-uri <span class="label label-warning">http://www.mbsim-env.de/MBXMLUtils</span> it is possible to embed a XML element multiple times. The full valid example syntax for this element is:</p>
<pre>&lt;pv:Embed href="file.xml" count="2+a" counterName="n" onlyif="n!=2"/&gt;</pre>
<p>or</p>
<pre>&lt;pv:Embed count="2+a" counterName="n" onlyif="n!=2"&gt;
  &lt;any_element_with_childs/&gt;
&lt;/pv:Embed&gt;
</pre>
<p>This will substitute the <span class="_element">&lt;pv:Embed&gt;</span> element in the current context <code>2+a</code> times with the element defined in the file <code>file.xml</code> or with <span class="_element">&lt;any_element_with_childs&gt;</span>. The insert elements have access to the global <a href="#parameters">parameters</a> and to a parameter named <code>n</code> which counts from <code>1</code> to <code>2+a</code> for each insert element. The new element is only insert if the octave expression defined by the attribute <span class="_attributeNoMargin">onlyif</span> evaluates to <code>1</code> (<code>true</code>). If the attribute <span class="_attributeNoMargin">onlyif</span> is not given it is allways <code>1</code> (<code>true</code>).<br/>
The attributes <span class="_attribure">count</span> and <span class="_attribure">counterName</span> must be given both or none of them. If none are given, then <code>count</code> is <code>1</code> and <span class="_attributeNoMargin">counterName</span> is not used.</p>

<p>The first child element of <span class="_element">&lt;pv:Embed&gt;</span> can be the element <span class="_element">&lt;pv:localParameter&gt;</span> which has one child element <a class="_element" href="#parameters">&lt;p:parameter&gt;</a> OR a attribute named <span class="_attributeNoMargin">href</span>. In this case the global parameters are expanded by the parameters given by the element <span class="_element">&lt;p:parameter&gt;</span> or in the file given by <code>href</code>. If a parameter already exist then the parameter is overwritten.</p>
<pre>&lt;pv:Embed count="2+a" counterName="n" onlyif="n!=2"&gt;
  &lt;p:Parameter xmlns:p="http://www.mbsim-env.de/MBXMLUtils"&gt;
    &lt;p:scalarParameter name="h1"&gt;0.5&lt;/p:scalarParameter&gt;
    &lt;p:scalarParameter name="h2"&gt;h1&lt;/p:scalarParameter&gt;
  &lt;/p:Parameter&gt;
  &lt;any_element_with_childs/&gt;
&lt;/pv:Embed&gt;
</pre>

    <h1><a id="measurements" href="#measurements-content">8 Measurements</a></h1>
    <p>The following subsections show all defined measurements.</p>
    <p>The column "Unit Name" in the tables is the name of the unit and the column
      "Conversion to SI Unit" is a expression which converts a value of this unit to the SI unit.</p>
    <p>The <span class="bg-success">highlighted row</span> shows the SI unit of this measurement which always has a conversion of "value".</p>
    <xsl:apply-templates select="/mm:measurement/mm:measure">
      <xsl:sort select="@name"/>
    </xsl:apply-templates>

    <hr/>
    <span class="pull-left small"><a href="/mbsim/html/impressum_disclaimer_datenschutz.html#impressum">Impressum</a> /
    <a href="/mbsim/html/impressum_disclaimer_datenschutz.html#disclaimer">Disclaimer</a> /
    <a href="/mbsim/html/impressum_disclaimer_datenschutz.html#datenschutz">Datenschutz</a></span><span class="pull-right small">
    Generated on <span class="DATETIME"><xsl:value-of select="$DATETIME"/></span> for MBXMLUtils by MBXMLUtils 
    <a href="/">Home</a>
    </span>
    </body></html>
  </xsl:template>

  <xsl:template match="/mm:measurement/mm:measure">
    <h2><a>
      <xsl:attribute name="id">
        <xsl:value-of select="@name"/>
      </xsl:attribute>
      <xsl:attribute name="href">#<xsl:value-of select="@name"/>-content</xsl:attribute>
      <code><xsl:value-of select="@name"/></code>
    </a></h2>
    <table class="table table-condensed table-striped table-hover">
      <thead>
        <tr> <th>Unit Name</th> <th>Conversion to SI Unit</th> </tr>
      </thead>
      <tbody>
        <xsl:apply-templates select="mm:unit"/>
      </tbody>
    </table>
  </xsl:template>

  <xsl:template match="mm:unit">
    <tr>
      <!-- mark the SI unit -->
      <xsl:if test="../@SIunit=@name">
        <xsl:attribute name="class">success</xsl:attribute>
      </xsl:if>
      <!-- unit name -->
      <td><code><xsl:value-of select="@name"/></code></td>
      <!-- outout conversion -->
      <td><code><xsl:value-of select="."/></code></td>
    </tr>
  </xsl:template>

</xsl:stylesheet>
