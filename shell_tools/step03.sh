#!/bin/bash

i="0"
res=""

echo output > output03
while true; 
do
res=$(bash failing.sh)
i=$[$i+1]
echo $res >> output03
if [[ $res == "Something went wrong" ]];
then
break
fi
done
echo "Steps: $i"
