echo "Configuring and building Spare Template based Reconstruction ..."

mkdir -p build
cd build
cmake ..
make -j 4