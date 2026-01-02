#!/bin/bash
set -e

if [ ! -d "submodules/dnd-5e-srd" ]; then
    git clone https://github.com/bagelbits/5e-srd-api.git submodules/dnd-5e-srd
fi

mkdir -p source/corpus
cp submodules/dnd-5e-srd/markdown/*.md source/corpus/

echo "✓ Скопировано файлов: $(ls -1 source/corpus/*.md | wc -l)"

