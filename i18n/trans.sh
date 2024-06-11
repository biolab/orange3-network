if [ "$#" -ne 1 ]
then
    echo "Usage: trans <language> <destination>"
else
    dest=$1
    trubar --conf trubar-config.yaml translate -s ../orangecontrib/network -d $dest/orangecontrib/network msgs.jaml
fi
