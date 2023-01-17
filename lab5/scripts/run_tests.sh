rm -rf out.txt
touch out.txt
cp "$1" "tests/solution"
cd tests
for (( t = 1; t <= $2; t++ ));
do
	a=`cat $t.in`
	if [[ ("$a" = "") ]];
	then
		continue
	fi
	./solution < "$t.in" >> ../out.txt
    # rm -rf "in.data" "out.data"
done
