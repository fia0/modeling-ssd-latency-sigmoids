#!/bin/env bash

set -e

. prep_fio.sh

BLOCK_SIZE=$((256 * 1024))
IO_DEPTH=256
RW_RATIO=1.0
NAME=probe
OUT="${NAME}_${BLOCK_SIZE}_${RW_RATIO}_${IO_DEPTH}"
OUT_JSON="${OUT}.json"

DIRECTORY=/vol1

"${FIO_DIR}/fio" \
    --name=probe \
    --readwrite=randrw \
    --rwmixread="$(echo "100 * $RW_RATIO" | bc)" \
    --rwmixwrite="$(echo "100 * (1.0 - $RW_RATIO)" | bc)" \
    --ioengine=io_uring \
    --iodepth=${IO_DEPTH} \
    --blocksize=${BLOCK_SIZE} \
    --size=10g \
    --runtime=60 \
    --time_based \
    --direct=1 \
    --directory="$DIRECTORY" \
    --group_reporting=1 \
    --output="$OUT_JSON" \
    --output-format='json+'

"${FIO_DIR}/tools/fio_jsonplus_clat2csv" "$OUT_JSON" "${OUT}.csv"
