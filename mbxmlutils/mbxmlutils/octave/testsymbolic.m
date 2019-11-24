try

format short e;

% init
fmatvec_symbolic_swig_octave;

% constructors
cs=3.1
is=IndependentVariable()
ss=SymbolicExpression(3.1)
cv=[3.1;3.1;3.1]
iv=VectorIndep(3,fmatvec_symbolic_swig_octave.NONINIT)
sv=VectorSym(3,fmatvec_symbolic_swig_octave.INIT,SymbolicExpression(3.1))
cm=[3.1,3.1,3.1;3.1,3.1,3.1;3.1,3.1,3.1]
sm=MatrixSym(3,3,fmatvec_symbolic_swig_octave.INIT,SymbolicExpression(3.1))

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
sin(is)
sin(ss)

% matrix/vector functions
cm'
sm'
norm(cv)
norm(iv)
norm(sv)

% horz-/vertcat
[cs;cs]
[cs;is]
[cs;ss]
[is;cs]
[is;is]
[is;ss]
[ss;cs]
[ss;is]
[ss;ss]
[cs,cs]
[cs,is]
[cs,ss]
[is,cs]
[is,is]
[is,ss]
[ss,cs]
[ss,is]
[ss,ss]
[cv,cv]
[cv,iv]
[cv,sv]
[iv,cv]
[iv,iv]
[iv,sv]
[sv,cv]
[sv,iv]
[sv,sv]
[MatrixSym(1,3);MatrixSym(1,3)]

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
