# course-video-to-doc
Extract ppt of online course video with intelligent split shot algorithm

用智能分镜头算法提取网上课程视频的ppt

### step-1 choose a good scence clip algorithm
git clone https://github.com/soCzech/TransNetV2.git

TransNet V2: Shot Boundary Detection Neural Network

download the network weights. Does file transnetv2-weights/saved_model.pb exist? It's on git-lfs so maybe git clone did not download it.

### step-2 build the environment
follow the TransNetV2 inference README and use the NVIDIA DOCKER.

docker build -t transnet -f inference/Dockerfile .

### step-3 video to doc
pip install img2pdf moviepy 

replace the  inference's code "transnetv2.py"

use "python transnetv2.py ../test.mp4 --threshold 0.5 --pdf --frame-type triple" to get a pdf

### TODO
1. suport set a box area in the video
2. faster pipeline
3. better scence clip algorithm

欢迎大家提出修改意见！