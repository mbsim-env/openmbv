try

% init
sym=swigLocalLoad('fmatvec_symbolic_swig_octave');

% constructors
cs=3.1
is=sym.IndependentVariable()
ss=sym.SymbolicExpression(3.1)
cv=[3.1;3.1]
iv=sym.VectorIndep(2,sym.NONINIT)
sv=sym.VectorSym(2,sym.INIT,sym.SymbolicExpression(3.1))
cm=[3.1,3.1;3.1,3.1]
sm=sym.MatrixSym(2,2,sym.INIT,sym.SymbolicExpression(3.1))

% operator *
cs*cs 
cs*is 
cs*ss 
cs*cv 
cs*iv 
cs*sv 
cs*cm 
cs*sm 
is*cs 
is*is 
is*ss 
is*cv 
is*iv 
is*sv 
is*cm 
is*sm 
ss*cs 
ss*is 
ss*ss 
ss*cv 
ss*iv 
ss*sv 
ss*cm 
ss*sm 
cv*cs 
cv*is 
cv*ss 
iv*cs 
iv*is 
iv*ss 
sv*cs 
sv*is 
sv*ss 
cm*cs 
cm*is 
cm*ss 
cm*cv 
cm*iv 
cm*sv 
cm*cm 
cm*sm 
sm*cs 
sm*is 
sm*ss 
sm*cv 
sm*iv 
sm*sv 
sm*cm 
sm*sm

% operator /
cs/cs
is/cs
ss/cs
cv/cs
iv/cs
sv/cs
cm/cs
sm/cs
cs/is
is/is
ss/is
cv/is
iv/is
sv/is
cm/is
sm/is
cs/ss
is/ss
ss/ss
cv/ss
iv/ss
sv/ss
cm/ss
sm/ss

% operator +
cs+cs
is+cs
ss+cs
cs+is
is+is
ss+is
cs+ss
is+ss
ss+ss
cv+cv
iv+cv
sv+cv
cv+iv
iv+iv
sv+iv
cv+sv
iv+sv
sv+sv
cm+cm
sm+cm
cm+sm
sm+sm

% operator -
cs-cs
is-cs
ss-cs
cs-is
is-is
ss-is
cs-ss
is-ss
ss-ss
cv-cv
iv-cv
sv-cv
cv-iv
iv-iv
sv-iv
cv-sv
iv-sv
sv-sv
cm-cm
sm-cm
cm-sm
sm-sm

% functions on symbolic scalars
sin(cs)
sin(cv)
sin(cm)
sym.sin(is)
sym.sin(ss)

% matrix/vector functions
cm'
sm'
norm(cv)
sym.norm(iv)
sym.norm(sv)

catch err

disp(err);
exit(1);

end
