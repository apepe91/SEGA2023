# SEGA2023
**MICCAI 2023 CHALLENGE SEG.A. - Segmentation of the Aorta**


This repository has everything you need to submit your algorithm to <b>SEG.A. 2023</b>.

All verified accounts on Grand Challenge who joined the SEG.A. challenge will be able to submit their algorithm as Docker container directly on the challenge website: https://multicenteraorta.grand-challenge.org/


If this is your first time, you might find the following documentation useful:

Here are some useful documentation links for your submission process:
- [How to create an algorithm container on Grand Challenge](https://grand-challenge.org/blogs/create-an-algorithm/)
- [Grand Challenge documentation](https://comic.github.io/grand-challenge.org/algorithms.html)
- [Docker documentation](https://docs.docker.com/)


## Prerequisites

You will need to have [Docker](https://docs.docker.com/) installed on your system. We recommend using Linux with a Docker installation. If you are on Windows, please use [WSL 2.0](https://docs.microsoft.com/en-us/windows/wsl/install).

## Prediction format

**Main task**: The algorithm reads the test cases from a user-defined folder and saves the AVT segmentation in another user-defined folder. For each CTA volume in the input folder "*ctaXX.nrrd*", the algorithm saves a segmentation file "*ctaXX.seg.nrrd*" in the output folder. 

**Subtask 1 (Optional)**: Additionally to the binary segmentation, the algorithm also generates a volumetric mesh. The volumetric mesh is saved in the same folder as the binary segmentation with filename "*ctaXX.msh*". 

**Subtask 2 (Optional)**: Additionally to the binary segmentation, the algorithm also generates a surface mesh. The surface mesh is saved in the same folder as the binary segmentation with filename "*ctaXX.stl*". The STL files will be used for the qualitative evaluation. 

## Adapting the container to your algorithm

1. First, clone this repository:

```
git clone https://github.com/...
```



If something does not work for you, please do not hesitate to [contact us](mailto:sega.grandchallenge@gmail.com) or [add a post in the forum](https://grand-challenge.org/forums/forum/segmentation-modeling-and-meshing-of-the-aortic-vessel-tree-694/). 

## Acknowledgments

The repository is greatly inspired and adapted from [SurgToolLoc](https://github.com/aneeqzia-isi/surgtoolloc2022-category-1) and [AutoImplant](https://autoimplant2021.grand-challenge.org/)



