while read line; do
    dload_loc="./downloads/${line#https://data.together.xyz/redpajama-data-1T/v1.0.0/}"
    mkdir -p $(dirname $dload_loc)
    # wget "$line" -O "$dload_loc"
    if [ ! -e "$dload_loc" ]; then
        wget "$line" -O "$dload_loc"
        echo "Downloaded: $line"
    else
        echo "File already exists, skipping download: $dload_loc"
    fi
done < ./new_urls/c4.txt