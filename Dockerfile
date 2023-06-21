FROM ubuntu:18.04
MAINTAINER Aarthi Ramakrishnan <aarthi.ramakrishnan@mssm.edu>

RUN apt-get update && apt-get install -y --force-yes \
	python3.8 \
	python3-pip \
	gcc \
	make \
	libbz2-dev \
	zlib1g-dev \
	libncurses5-dev \
	libncursesw5-dev \
	liblzma-dev \
	wget \
	unzip \
	bedtools

RUN pip3 install Cython==0.29.24
RUN pip3 install numpy==1.19.5
RUN pip3 install pandas==1.1.5
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade Pillow
RUN pip3 install matplotlib==3.3.3
RUN pip3 install -U scikit-learn==0.24.2
RUN pip3 install PyYAML==5.4.1
RUN pip3 install torch==1.9.0
RUN python3 -m pip install scipy==1.5.4
RUN pip3 install pysam==0.16.0.1
RUN pip3 install pybedtools==0.8.2
RUN pip3 install tensorboard

# Install featureCounts
RUN wget https://sourceforge.net/projects/subread/files/subread-2.0.3/subread-2.0.3-Linux-x86_64.tar.gz
RUN tar xzvf subread-2.0.3-Linux-x86_64.tar.gz

# Install DeepRegFinder
RUN wget -P ~/ https://github.com/shenlab-sinai/DeepRegFinder/archive/refs/heads/master.zip
WORKDIR /root
RUN unzip master.zip
WORKDIR /root/DeepRegFinder-master
ENV PATH="${PATH}:/root/DeepRegFinder-master"

ENV PATH="/subread-2.0.3-Linux-x86_64/bin:/DeepRegFinder/DeepRegFinder:${PATH}"
