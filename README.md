# Face Recognition

A light weight face recognition implementation using a pre-trained facenet model. Most of the code is taken from David Sandberg's  [facenet](https://github.com/davidsandberg/facenet) repository.

## Steps to follow:
1. Create a dataset of faces for each person and arrange them in below order
```
root folder  
│
└───Person 1
│   │───IMG1
│   │───IMG2
│   │   ....
└───Person 2
|   │───IMG1
|   │───IMG2
|   |   ....
```
2. Align the faces using MTCNN or dllib. Please use the scripts available in lib/src/align. For this project i aligned faces using MTCNN.(Please have a look at [aligning faces](https://github.com/davidsandberg/facenet/wiki/Validate-on-lfw#4-align-the-lfw-dataset) if you need any clarifications) 

[Before alignment]<img src="https://github.com/vinayakkailas/face_recognition/blob/master/server/static/images/1.jpg"  width="250" height="250" />    [After alignment] <img src="https://github.com/vinayakkailas/face_recognition/blob/master/server/static/images/2.png"  width="250" height="250" /> 

3. Download [pre-trained-weight](https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz) ,extract and keep it in lib/src/ckpt folder (for detailed info about availabe weights: [available-weights](https://github.com/davidsandberg/facenet#pre-trained-models)) 
4. Create face embeddings using pre-trained facenet model. Run the below scripts by changing the folder paths.(edit paths in [lines](https://github.com/vinayakkailas/face_recognition/blob/49ed6e80a4205e6a8fa1a18dbdc8976d4be29535/lib/src/create_face_embeddings.py#L49))
```
  python lib/src/create_face_embeddings.py 
 ```
  Once you run the script succesfully, a pickle file with name face_embeddings.pickle will be generated inside lib/src folder
 
5. Start the server by running the command
```
  python server/rest-server.py
```
  access the UI using url https://127.0.0.1:5000. It will show a button with face recognition as label. Once you click on it,      automatically your primary camera will get turned on and start recognizing the faces.
 
 ## sample result 
 
 ![alt text](https://github.com/vinayakkailas/face_recognition/blob/master/server/static/images/vinayak.jpeg)
 
for more information, please go through my [blog](https://medium.com/@vinayakvarrier/building-a-real-time-face-recognition-system-using-pre-trained-facenet-model-f1a277a06947)  

NOTE: The faces are identified using retrievel method, instead if you have enough data, you can train a classifier on top of face embeddings ([Train-a-classifier-on-own-images](https://github.com/davidsandberg/facenet/wiki/Train-a-classifier-on-own-images)) 

References:

* Deepface paper https://www.cs.toronto.edu/~ranzato/publications/taigman_cvpr14.pdf
* Facenet paper https://arxiv.org/pdf/1503.03832.pdf
* Pre-trained facenet model https://github.com/davidsandberg/facenet
