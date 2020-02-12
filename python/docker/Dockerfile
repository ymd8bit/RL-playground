FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ENV LC_ALL='C' 
ENV DEBIAN_FRONTEND noninteractive 

ARG workspace
ENV workspace ${workspace}
WORKDIR $workspace

RUN apt-get update && \
  apt-get install --no-install-recommends -y python3 python3-dev python3-dbg python3-pip python3-setuptools python3-tk \
  git vim wget cmake ninja-build build-essential

RUN apt-get install --no-install-recommends -y xvfb x11vnc python-opengl icewm xterm

COPY requirements.txt $workspace/
RUN python3 -m pip install -r requirements.txt

# for VSCode
RUN apt-get install --no-install-recommends -y \
  exuberant-ctags

COPY requirements.txt $workspace/
COPY run.sh /
RUN chmod +x /run.sh

CMD ["/run.sh"]
