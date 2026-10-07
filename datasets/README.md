# Dataset preparation

Images and masks are not distributed in this repository. Obtain the datasets from their respective authors, then place each dataset's PNG images and binary masks under `images/` and `masks/`.

The checked local data root is `D:\document\ADGFNet\datasets`. Use `--dataset_dir` to read it without copying the image data into this repository.

The included `img_idx` lists are copied from the supplied local experiments. Each identifier is the PNG filename stem and must exist in both `images` and `masks`.

| Dataset | Train | Test | Total |
| --- | ---: | ---: | ---: |
| NUDT-SIRST | 663 | 664 | 1327 |
| IRSTD-1K | 800 | 201 | 1001 |

No duplicate identifiers, training/test overlap, or missing image/mask files were found in these lists. The IRSTD-1K total differs from the manuscript's description of 1000 images; preserve the experimental lists when comparing the bundled checkpoint results.

Masks must use 0/255 values. Images are converted to grayscale and normalized using the constants in `utils/utils2.py`. NUDT-SIRST training crops are 256 × 256; IRSTD-1K crops are 512 × 512. At test time, full images are padded to a multiple of 32 and predictions are cropped back to the original size.

The supplied IRSTD-1K data has 72 training and 13 test masks containing intermediate grayscale boundary values. The loader consistently binarizes all masks using `raw_mask > 127.5` (equivalent to threshold 0.5 after division by 255), for both training and evaluation. This affects the labels seen by the model/evaluator, without rewriting the original PNG files.

Dataset redistribution terms remain those of the original authors. A public download link for the prepared local data has not been supplied.
