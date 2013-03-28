cat ../levantine_corpus_gansell.csv | while read l
do 
    id=`echo $l | cut -d, -f1`
    name=`echo $l | cut -d, -f2`
    mv "$id.jpg" "$name.jpg"
    touch "$name.jpg"
done
