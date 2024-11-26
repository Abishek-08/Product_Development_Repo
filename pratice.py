from deepface import DeepFace

models = [
  "VGG-Face", 
  "Facenet", 
  "Facenet512", 
  "OpenFace", 
  "DeepFace", 
  "DeepID", 
  "ArcFace", 
  "Dlib", 
  "SFace",
  "GhostFaceNet",
]


# result = DeepFace.verify(
#   img1_path = "./images/Abishek K/abi.jpg",
#   img2_path = "./images/Abishek K/12120.jpg",
#   model_name = models[2]
# )

#face recognition
dfs = DeepFace.find(
  img_path = "vicky.jpg",
  db_path = "D:\Product Development\Praticing_Folder\Face_Deep\images", 
  model_name = models[2],
)

print("default: ",dfs)

print("default0: ",dfs[0])

print("testIdentity: ",dfs[0]['identity'])

print("test: ",dfs[0]['identity'][1])

#To get the name of the frame
print("DFS: ",dfs[0]['identity'][0].split('\\')[5].split('.')[0])

#To get the coordinate of the frame
print("FrameX1: ",dfs[0]['source_x'][0])

# print(result)

# for result in dfs:
#     #To get the name of the frame
#     print("DFS: ",result['identity'][0].split('\\')[4].split('.')[0])

