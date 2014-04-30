cat ../levantine_corpus_gansell.csv | awk '{printf("%d,%s\n", NR-1, $0)}' | while read l
do 
    id=`echo $l | cut -d, -f1`
    name=`echo $l | cut -d, -f2`
    mv "$id.jpg" "$name.jpg"
    if [ -e "$name.jpg" ]
    then
        touch "$name.jpg"
    fi
done
