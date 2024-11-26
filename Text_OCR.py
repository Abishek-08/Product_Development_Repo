from doctr.models import ocr_predictor
from doctr.io import DocumentFile
import matplotlib.pyplot as plt

model = ocr_predictor(pretrained=True,det_arch="db_resnet50",reco_arch="crnn_vgg16_bn")

image = DocumentFile.from_images('./data/table1.png')

result = model(image)

result.show()

json_formart = result.export()
print(json_formart)
text_output = result.render()
print(text_output)


#Rebuild the page
synthetic_pages = result.synthesize()
plt.imshow(synthetic_pages[0]); plt.axis('off'); plt.show()

