% check octave version
myversion=version;
if str2num(myversion(1:index(myversion,'.')-1)) <= 2
  octave2=1;
else
  octave2=0;
end

myargv=argv;
FILENAME=myargv{1};
% open file
f=fopen(FILENAME,'rt');
nr=0;
while(1)
  % read on line of file
  line=fgetl(f);
  nr=nr+1;
  if octave2
    fid=fopen('.line', 'w'); fprintf(fid, '%s', line); fclose(fid);
    [dummy,ret]=system('grep -E "@(SCALAR|VECTOR|MATRIX|SCALARINTEGER)EXPRESSION{@" < .line > /dev/null');
    if ret==0, match=1; else match=[]; end
  else
    match=regexp(line, '@(SCALAR|VECTOR|MATRIX|SCALARINTEGER)EXPRESSION{@');
  end
  if size(match)(1)==0 | size(match)(2)==0
    % if nothing to evaluate, print line
    printf('%s\n',line);
  else
    % get expression to evaluate (and pre, post-expression)
    if octave2
      preexp=system('sed -re "s/(.*)@(SCALAR|VECTOR|MATRIX|SCALARINTEGER)EXPRESSION\\{@(.*)@\\}@(.*)/\\1/" < .line');
      type=system('sed -re "s/(.*)@(SCALAR|VECTOR|MATRIX|SCALARINTEGER)EXPRESSION\\{@(.*)@\\}@(.*)/\\2/" < .line');
      exp=system('sed -re "s/(.*)@(SCALAR|VECTOR|MATRIX|SCALARINTEGER)EXPRESSION\\{@(.*)@\\}@(.*)/\\3/" < .line');
      postexp=system('sed -re "s/(.*)@(SCALAR|VECTOR|MATRIX|SCALARINTEGER)EXPRESSION\\{@(.*)@\\}@(.*)/\\4/" < .line');
    else
      [aa,bb,cc,dd,ee,ff]=regexp(line, '(.*)@(SCALAR|VECTOR|MATRIX|SCALARINTEGER)EXPRESSION{@(.*)@}@(.*)');
      preexp=ee{1}{1};
      type=ee{1}{2};
      exp=ee{1}{3};
      postexp=ee{1}{4};
    end
    % evaluate expression
    val=eval(exp);
    if strcmp(type,'SCALAR')==1 || strcmp(type,'SCALARINTEGER')==1
      % if scalar, check size
      if sum(size(val)!=[1,1])
        printf('%s:%d: ERROR: Scalar expected, but got value of size %dx%d\n',FILENAME,nr,size(val));
        exit;
      end
      % if scalarinteger, check for integer
      if strcmp(type,'SCALARINTEGER')==1 && round(val)!=val
        printf('%s:%d: ERROR: Scalar integer expected, but got %e\n',FILENAME,nr,val);
        exit;
      end
      % if scalar print evaluated line
      if strcmp(type,'SCALARINTEGER')!=1
        printf('%s%.15e%s\n', preexp, val, postexp);
      else
        printf('%s%d%s\n', preexp, val, postexp);
      end
    else
      % if vector, check size
      if strcmp(type,'VECTOR')==1 && size(val,2)!=1
        printf('%s:%d: ERROR: Column vector expected, but got value of size %dx%d\n',FILENAME,nr,size(val));
        exit;
      end
      % if vector or matrix print evaluated line
      printf('%s[ ', preexp);
      for r=1:size(val,1)
        for c=1:size(val,2)
          printf('%.15e',val(r,c));
          if c!=size(val,2), printf(' , '); end
        end
        if r!=size(val,1), printf(' ; '); end
      end
      printf(' ]%s\n', postexp);
    end
  end
  if feof(f), break; end
end
