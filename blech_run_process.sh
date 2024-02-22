DIR=$1
echo "Processing $DIR"
for i in {1..10}; do
    echo Retry $i
    bash $DIR/temp/blech_process_parallel.sh
done
