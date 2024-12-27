AFAD-Dataset
===

https://afad-dataset.github.io/

Directory Hierarchy
---

```
age/gender/image

gender: integer, where 111 stands for "male", 112 for "female".
```

Tips on Statistics
---

AFAD-FULL:

```
# Total number of .jpg files
$ find . -type f -name '*.jpg' | wc -l
165501

# Number of photos for male
$ find . -type f -path '*/111/*' | wc -l
101526

# Number of photos for female
$ find . -type f -path '*/112/*' | wc -l
63989
```

AFAD-Lite:

```
$ find . -type f -name '*.jpg' | wc -l
59344

$ find . -type f -path '*/111/*' | wc -l
34817

$ find . -type f -path '*/112/*' | wc -l
24527
```

Citation
---

Zhenxing Niu, Mo Zhou, Le Wang, Xinbo Gao, Gang Hua.
Ordinal Regression with a Multiple Output CNN for Age Estimation.
CVPR, 2016.

Notice
---

This dataset is for academic use only.
