# CSCI5563-Project--3D-Reconstruction
3D Reconstruction of Massively Occluded Faces

Baseline: original baseline code and our attempts to run the whole thing. Used pretrained model weights and provided code (testBatchModel.py with some small modifications) for facial feature detection and general 3D shape generation

GenerateBumpMaps-attempt1: Method original baseline cited for generating the bump maps for their training data. Written in Matlab. Couldn't get this working fully because our depth data wasn't aligned properly. Ended up using GIMP instead.

GenerativeInpainting-attempt1: Original inpainting method we proposed using in the proposal. Couldn't get this compiling as is so found a slightly different method proposed in a different venue (see report for details).

TrainingData: 6,000 images from the CelebA-HQ dataset we used for training our new model.

new_bump_map_inpainting.py: Our depth inpainting network we developed. We ran it all in Google Colab and hosted the data in Google Drive but I copied all the code here. The oiginal filepaths are still in there though. Follow this link to see Colab version: https://colab.research.google.com/drive/1ulu1Tyw7FvzS0HIhj7LlCArTpS0TRxWX?usp=sharing
