FILENAME=argv{1};
% open file
f=fopen(FILENAME,'rt');
nr=0;
while(1)
  % read on line of file
  line=fgetl(f);
  nr=nr+1;
  match=regexp('@(SCALAR|VECTOR|MATRIX)EXPRESSION\\{@',line);
  if size(match)==[0,0]
    % if nothing to evaluate, print line
    printf('%s\n',line);
  else
    % get expression to evaluate (and pre, post-expression)
    [match,preexp,type,exp,postexp]=regexp('(.*)@(SCALAR|VECTOR|MATRIX)EXPRESSION\\{@(.*)@\\}@(.*)',line);
    % evaluate expression
    val=eval(exp);
    if type=='SCALAR'
      % if scalar, check size
      if sum(size(val)!=[1,1])
        printf('%s:%d: ERROR: Scalar expected, but got value of size %dx%d\n',FILENAME,nr,size(val));
        exit;
      end
      % if scalar print evaluated line
      printf('%s%.15e%s\n', preexp, val, postexp);
    else
      % if vector, check size
      if type=='VECTOR' && size(val,2)!=1
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
