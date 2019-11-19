try

% init
sym=swigLocalLoad('fmatvec_symbolic_swig_octave');

% constructors
cs=3.1
is=sym.IndependentVariable()
ss=sym.SymbolicExpression(3.1)
cv=[3.1;3.1;3.1]
iv=sym.VectorIndep(3,sym.NONINIT)
sv=sym.VectorSym(3,sym.INIT,sym.SymbolicExpression(3.1))
cm=[3.1,3.1,3.1;3.1,3.1,3.1;3.1,3.1,3.1]
sm=sym.MatrixSym(3,3,sym.INIT,sym.SymbolicExpression(3.1))

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

% horz-/vertcat
vertcat(cs,cs)
vertcat(cs,is)
vertcat(cs,ss)
vertcat(is,cs)
vertcat(is,is)
vertcat(is,ss)
vertcat(ss,cs)
vertcat(ss,is)
vertcat(ss,ss)
horzcat(cs,cs)
horzcat(cs,is)
horzcat(cs,ss)
horzcat(is,cs)
horzcat(is,is)
horzcat(is,ss)
horzcat(ss,cs)
horzcat(ss,is)
horzcat(ss,ss)
horzcat(cv,cv)
horzcat(cv,iv)
horzcat(cv,sv)
horzcat(iv,cv)
horzcat(iv,iv)
horzcat(iv,sv)
horzcat(sv,cv)
horzcat(sv,iv)
horzcat(sv,sv)
vertcat(sym.MatrixSym(1,3),sym.MatrixSym(1,3))

% helper functions
rotateAboutX(cs)
rotateAboutY(cs)
rotateAboutZ(cs)
cardan(cs, cs*2, cs*3)
euler(cs, cs*2, cs*3)
cardan(cv)
euler(cv)
tilde(cv)
tilde(cm)
rotateAboutX(is)
rotateAboutY(is)
rotateAboutZ(is)
cardan(is, is*2, is*3)
euler(is, is*2, is*3)
cardan(iv)
euler(iv)
tilde(iv)
rotateAboutX(ss)
rotateAboutY(ss)
rotateAboutZ(ss)
cardan(ss, ss*2, ss*3)
euler(ss, ss*2, ss*3)
cardan(sv)
euler(sv)
tilde(sv)
tilde(sv)

catch err

disp(err);
exit(1);

end
