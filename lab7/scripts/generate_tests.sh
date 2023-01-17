cd ./tests
MAX_TEST=99
for (( t = 1; t <= $MAX_TEST; t++ ));
do
	rm -rf "$t.in" "$t.out"
done
for (( t = 1; t <= $1; t++ ));
do
	touch "$t.in" "$t.out"
done
