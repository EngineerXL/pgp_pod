cp solution tests/solution
cd scripts
make checker
cp checker ../tests/checker
cd ../tests
for (( t = 1; t <= $1; t++ ));
do
	a=`cat $t.in`
	if [[ ("$a" = "") ]];
	then
		continue
	fi
	touch out.txt verdict.txt
	mpirun -np $2 ./solution < "$t.in"
    ./checker "$t.out" "out.txt" > verdict.txt
	a=`cat verdict.txt`
	if [[ ("$a" = "") ]];
	then
		echo "Test $t OK"
	else
		echo "Test $t WA"
        echo $a
		break
	fi
	rm -rf out.txt verdict.txt
done
