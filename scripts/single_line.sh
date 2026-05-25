for file in *.fasta; do
    base="${file%.fasta}"

    awk '/^>/ {printf("\n%s\n",$0);next;} {printf("%s",$0);} END {printf("\n");}' "$file" \
    | sed '/^$/d' > "${base}_1.fasta"

    rm "$file"
done
