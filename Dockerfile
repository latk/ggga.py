FROM python:3.7-alpine
MAINTAINER Lukas Atkinson <opensource@LukasAtkinson.de>

# Adapted from https://github.com/abn/scipy-docker-alpine
#     and from https://github.com/o76923/alpine-numpy-stack

RUN apk --update --no-cache add --virtual=.scipy-runtime \
        build-base libgfortran openblas freetype libpng tcl tk

RUN apk --update --no-cache add --virtual=.scipy-build \
        gfortran musl-dev pkgconf openblas-dev freetype-dev \
    && ln -s /usr/include/locale.h /usr/include/xlocale.h \
    && pip3 install --no-cache-dir numpy \
    && pip3 install --no-cache-dir matplotlib \
    && pip3 install --no-cache-dir scipy \
    && pip3 install --no-cache-dir pandas \
    && rm /usr/include/xlocale.h \
    && apk del .scipy-build

# Need to install sklearn from source rather than PyPI
# because the package contains pre-generated cython files
# that are incompatible with Python 3.7
RUN pip3 install --no-cache-dir -U cython \
    && pip3 install --no-cache-dir https://github.com/scikit-learn/scikit-learn/archive/0.20.1.tar.gz

WORKDIR /ggga

COPY ./requirements.txt ./requirements-dev.txt ./
RUN pip3 --no-cache-dir install -r ./requirements.txt -r ./requirements-dev.txt

COPY . ./
RUN pip3 --no-cache-dir install .

ENTRYPOINT ["python3", "-m", "ggga.examples"]
