sudo apt-get update
sudo apt install python3-pil.imagetk -y
sudo apt install python3-tk -y

cp GigE-V-Framework_aarch64_2.21.1.0195.tar.gz $HOME
cd $HOME
tar -zxf GigE-V-Framework_aarch64_2.21.1.0195.tar.gz

cd DALSA
./corinstall
