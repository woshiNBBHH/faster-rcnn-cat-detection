# Cat/Dog 53,337-image dataset

This directory is intentionally separate from the original `datastes/` cat-only dataset.

Expected final structure:

```text
cat_dog_dataset_53337/
├── train/  # image + same-stem Labelme JSON
├── val/
├── test/
└── split_summary.json
```

The source images are not committed here until a dataset hosting solution with enough
capacity is selected. The source collection is about 13 GB, which exceeds the included
10 GiB Git LFS storage on a GitHub Free/Pro account.
