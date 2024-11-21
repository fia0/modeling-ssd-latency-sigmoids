#!/bin/env bash

set -e

export FIO_RELEASE=3.33
export FIO_ARCHIVE=fio-${FIO_RELEASE}.tar.gz
export FIO_DIR=fio-fio-${FIO_RELEASE}
export FIO_URL=https://github.com/axboe/fio/archive/refs/tags/${FIO_ARCHIVE}

setup_fio() {
    wget ${FIO_URL}
    sha256sum -c checksum.txt
    tar xzf ${FIO_ARCHIVE}
    rm ${FIO_ARCHIVE}
    cd fio-fio-${FIO_RELEASE} && ./configure && make -j $(nproc)
}

if [ ! -d "${FIO_DIR}" ]; then
    setup_fio
fi
