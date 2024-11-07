#!/bin/env bash

set -e

FIO_RELEASE=3.33
FIO_ARCHIVE=fio-${FIO_RELEASE}.tar.gz
FIO_DIR=fio-fio-${FIO_RELEASE}
FIO_URL=https://github.com/axboe/fio/archive/refs/tags/${FIO_ARCHIVE}

BLOCK_SIZE=$((256 * 1024))
IO_DEPTH=256
RW_RATIO=1.0
NAME=probe
OUT="${NAME}_${BLOCK_SIZE}_${RW_RATIO}_${IO_DEPTH}"
OUT_JSON="${OUT}.json"

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

"${FIO_DIR}/fio" \
    --name=probe \
    --readwrite=randrw \
    --rwmixread="$(echo \"100 * $RW_RATIO\" | bc)" \
    --rwmixwrite="$(echo \"100 * \(1.0 - $RW_RATIO\)\" | bc)" \
    --ioengine=io_uring \
    --iodepth=${IO_DEPTH} \
    --blocksize=${BLOCK_SIZE} \
    --size=10g \
    --direct=1 \
    --directory=/vol1 \
    --group_reporting=1 \
    --output="$OUT_JSON" \
    --output-format='json+'

"${FIO_DIR}/tools/fio_jsonplus_clat2csv" "$OUT_JSON" "$OUT"
