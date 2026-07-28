# Example data

Twenty synthetic Chinese reviews written for this repository, showing the expected
TSV format: a `label<TAB>text_a` header, then one `label<TAB>text` row per line
(`1` positive, `0` negative). Split as train 12 / dev 4 / test 4.

These rows are made up and are not drawn from any dataset. They are for checking
the format and smoke-testing the scripts only; they are far too small to train a
usable model. Download the real datasets as described in the top-level README.
