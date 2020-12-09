FROM nvidia/cuda:10.2-base
CMD nvidia-smi
FROM gfootball
RUN apt-get update && apt-get install -y build-essential
RUN apt-get --purge remove -y nvidia*


RUN pip install --upgrade pip && \
	pip install torch torchvision
RUN pip3 install jupyter
COPY . /src/notebooks
WORKDIR /src/notebooks
ENV TINI_VERSION v0.6.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/bin/tini
RUN chmod +x /usr/bin/tini
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
