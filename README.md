# OptiDiff - Optical-map structural variation detection tool

## Requirements:

Depends on [OptiScan](https://gitlab.com/akdel/OptiScan)

## Installation

```shell script
git clone https://gitlab.com/akdel/optidiff.git
cd optidiff
pip install .
```

## Test with toy data
```shell script
OptiDiff data/genomes/yeast.cmap data/yeast10mb.fasta.bnx data/SVs_del/511699-520213-del.fasta
```