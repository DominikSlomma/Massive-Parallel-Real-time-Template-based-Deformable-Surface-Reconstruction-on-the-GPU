echo "Configuring and building of this repo ..."

mkdir -p build
cd build
cmake ..
make -j 4