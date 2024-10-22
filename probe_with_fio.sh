#!/bin/env bash

set -e

FIO_RELEASE=3.33
FIO_ARCHIVE=fio-${FIO_RELEASE}.tar.gz
FIO_DIR=fio-fio-${FIO_RELEASE}
FIO_URL=https://github.com/axboe/fio/archive/refs/tags/${FIO_ARCHIVE}

setup_fio() {
    wget ${FIO_URL}
    sha256sum -c checksum.txt
    tar xzf ${FIO_ARCHIVE}
    ${RM} ${FIO_ARCHIVE}
    cd fio-fio-${FIO_RELEASE} && ./configure && make
}

if [ ! -d "${FIO_DIR}" ]; then
    setup_fio
fi

"${FIO_DIR}/fio" \
    --name=probe \
    --readwrite=read \
    --ioengine=posixaio \
    --iodepth=256 \
    --blocksize=$((256 * 1024)) \
    --write_hist_log=probe \
    --size=160g \
    --direct=1 \
    --directory=/vol1 \
    --log_hist_msec=1 \
    --output=probe.json \
    --output-format='json+'
