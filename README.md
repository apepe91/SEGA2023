# SEGA2023
**MICCAI 2023 CHALLENGE SEG.A. - Segmentation of the Aorta**


This repository has everything you need to submit your algorithm to <b>[SEG.A. 2023](https://multicenteraorta.grand-challenge.org/)</b>.

All verified accounts on Grand Challenge who joined the SEG.A. challenge will be able to submit their algorithm as Docker container directly on the challenge website: https://multicenteraorta.grand-challenge.org/


If this is your first time, you might find the following documentation useful:

Here are some useful documentation links for your submission process:
- [How to create an algorithm container on Grand Challenge](https://grand-challenge.org/blogs/create-an-algorithm/)
- [Grand Challenge documentation](https://comic.github.io/grand-challenge.org/algorithms.html)
- [Docker documentation](https://docs.docker.com/)


## Prerequisites

You will need to have [Docker](https://docs.docker.com/) installed on your system. We recommend using Linux with a Docker installation. If you are on Windows, please use [WSL 2.0](https://docs.microsoft.com/en-us/windows/wsl/install).

## Adapting the container to your algorithm
A template for the submission container is available at [https://github.com/apepe91/SEGA2023/tree/main/SegaAlgorithm](https://github.com/apepe91/SEGA2023/tree/main/SegaAlgorithm)


## Prediction format

**Main task**: Binary segmentation mask.

**Subtask 1 (Optional)**: Additionally to the binary segmentation, the algorithm also generates a surface mesh in the form of an OBJ file. The surface mesh will be qualitatively assessed and ranked by field specialists. If you do not wish to participate to this subtask, submit a small cube mesh (trimesh.primitives.Box()) as shown the template. 

**Subtask 2 (Optional)**: Additionally to the binary segmentation, the algorithm also generates a surface mesh in the form of an OBJ file. This additional surface mesh will be quantitatively assessed for the creation of a volumetric mesh. If you do not wish to participate to this subtask, submit a small cube mesh (trimesh.primitives.Box()) as shown the template. 




If something does not work for you, please do not hesitate to [contact us](mailto:sega.grandchallenge@gmail.com) or [add a post in the forum](https://grand-challenge.org/forums/forum/segmentation-modeling-and-meshing-of-the-aortic-vessel-tree-694/). 

## Acknowledgments

The repository is greatly inspired and adapted from [SurgToolLoc](https://github.com/aneeqzia-isi/surgtoolloc2022-category-1) and [AutoImplant](https://autoimplant2021.grand-challenge.org/)



