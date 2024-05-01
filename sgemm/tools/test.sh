#!/bin/bash
# 1. build if bin file is not exist
if [ ! -f "build/main" ]; then
    mkdir -p build && cd build
    cmake ..
    make
    if [ $? -ne 0 ]; then
        echo "compile error"
        exit 1
    fi
    cd --
fi

# 2. test
rm -rf test
mkdir -p test
for((i=0;i<=1;i++)); do
    echo -n "test kernel: ${i}..."
    file_name="./test/test_kernel_${i}.log"
    ./build/main ${i} > ${file_name}
    if [ $? -ne 0 ]; then
        echo "kernel${i} error"
        exit 1
    fi

    # 3. if not cuBLAs, plot and save to images/
    if [ ${i} -gt 0 ]; then
        echo -n "Done. Ploting..."
        python3 tools/plot.py 0 ${i}
    fi
    echo "Done."
done
