python3 -m venv corsound_venv
source corsound_venv/bin/activate
pip3 install -r requirements.txt
ipython kernel install --user --name=corsound_venv
sudo apt install ffmpeg youtube-dl
wget https://www.robots.ox.ac.uk/~vgg/data/voxceleb/data/vox2_test_txt.zip
unzip vox2_test_txt.zip
cp .//extract_segment.py.fix .//corsound_venv/lib/python3.9/site-packages/voxceleb_luigi/extract_segment.py
#dev dataset (~6000 voices)
# wget http://www.robots.ox.ac.uk/\~vgg/data/voxceleb/data/vox2_dev_txt.zip
#unzip vox2_dev_txt.zip
