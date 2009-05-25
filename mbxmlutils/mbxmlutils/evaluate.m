myargv=argv;
INFILENAME=myargv{1};
OUTFILENAME=myargv{2};
% open files
fin=fopen(INFILENAME,'rt');
fout=fopen(OUTFILENAME,'wt');
while(1)
  % read one line of file
  line=fgetl(fin);

  % search ATTR
  while 1
    [S,bb,cc,dd,T]=regexp(line, '@ATTRB@(.+?)@ATTRE@');
    if length(S)==0, break; end
    attr=T{1}{1};
    % subst all {.*?}
    while 1
      [S,bb,cc,dd,T]=regexp(attr, '{(.+?)}');
      if length(S)==0, break; end
      rep=eval(T{1}{1});
      if round(rep)!=rep, error([T{1}{1} ' dose not evaluate to a integer in attribute!']); end
      attr=regexprep(attr, '{.+?}', sprintf('%d',rep), 'once');
    end
    line=regexprep(line, '@ATTRB@.+?@ATTRE@', sprintf('%s',attr), 'once');
  end
  % search TEXT
  while 1
    [S,bb,cc,dd,T]=regexp(line, '@TEXTB@(.+?)@TEXTE@');
    if length(S)==0, break; end
    val=eval(T{1}{1});
    if size(val,1)==1 && size(val,2)==1
      text=sprintf('%.15e', val);
    else
      text=sprintf('[');
      for r=1:size(val,1)
        for c=1:size(val,2)
          text=[text sprintf('%.15e',val(r,c))];
          if c!=size(val,2), text=[text sprintf(',')]; end
        end
        if r!=size(val,1), text=[text sprintf(';')]; end
      end
      text=[text sprintf(']')];
    end
    line=regexprep(line, '@TEXTB@.+?@TEXTE@', sprintf('%s',text), 'once');
  end

  fprintf(fout,'%s\n', line);
  if feof(fin), break; end
end
