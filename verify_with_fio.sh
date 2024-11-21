#!/bin/env bash

set -e

. prep_fio.sh

RW_RATIO=1.0
REQS_PER_BATCH=100
IO_DEPTH=128
ITERATION=100
INTERVAL=5s
SIZE=800
SEED=54321
BLOCK_SIZE=$((4 * 1024 * 1024))

"${FIO_DIR}/fio" \
    --name=verify \
    --readwrite=randrw \
    --rwmixread="$(echo "100 * $RW_RATIO" | bc)" \
    --rwmixwrite="$(echo "100 * (1.0 - $RW_RATIO)" | bc)" \
    --ioengine=io_uring \
    --iodepth=${IO_DEPTH} \
    --iodepth_batch_submit=$REQS_PER_BATCH \
    --blocksize=${BLOCK_SIZE} \
    --size=$((SIZE * BLOCK_SIZE)) \
    --io_size=$((REQS_PER_BATCH * ITERATION * BLOCK_SIZE)) \
    --thinktime= $INTERVAL \
    --direct=1 \
    --directory="$DIRECTORY" \
    --group_reporting=1 \
    --output="$OUT_JSON" \
    --output-format='json+'
