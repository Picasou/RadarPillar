
#!/bin/bash

if [ $# -eq 0 ];then
  if [ $GOLDEN_PATH ];then
    GOLDEN_PATH=${GOLDEN_PATH}
  else
    echo "param error! pls set GOLDEN_PATH env or use: ./diff.sh {golden_path}"
    echo "./diff.sh {golden_path}"
    exit 1
  fi
elif [ $# -eq 1 ];then
    GOLDEN_PATH=$1
else
  echo "param error! pls set GOLDEN_PATH env or use: ./diff.sh {golden_path}"
  exit 1
fi

#set -x

if_same=true
has_match_file=false
for file in ./*
do
    filename=$(basename $file )
    golden_file=${GOLDEN_PATH}/${filename}
    if [ ! -f $gloden_file ]
    then
      file_size="$(wc -c <"$file")"
      gloden_file_size="$(wc -c <"$gloden_file")"
      if [ $file_size -ne $gloden_file_size ]
      then
        echo "The size of $filename not match!"
      else
        diff ${file} ${golden_file} 1>diff.txt 2>&1 && result=0 || result=1

        if [ "$result" == 1 ];then
            echo "$filename is diff"
            if_same=false
        else
            has_match_file=true    
        fi
      fi
    #else
     # echo "$golden_file does not exist, please check filename."
    fi
done

if [ "$if_same" = true -a "$has_match_file" = true ]; then
  echo "All same!"
fi

