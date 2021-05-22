import torch
from fastapi import FastAPI, File, UploadFile
from efficientnet_pytorch import EfficientNet
from torch.autograd import Variable
from torch._C import device
from torchvision import transforms
from PIL import Image
import io
import json
from utils import class_names

# creates a basic fastapi app
app = FastAPI(
    title="Fitnest Model Deployment",
    version="0.1.0",
)


# @app.on_event("startup")
# def model_init():
# if torch.cuda.is_available():
#     device = torch.device("cuda")
# else:
device = torch.device("cpu")
# print(device.type)

weights_path = "./weights.pth"
model = EfficientNet.from_pretrained(
    "efficientnet-b2", weights_path=weights_path, num_classes=40
)
# model = torch.nn.DataParallel(model)
model.load_state_dict(torch.load(weights_path, map_location="cpu"))
model.to(device)
model.eval()

@app.post("/upload")
async def predict(image: UploadFile = File(...)):
    with open("data.json") as f:
      data = json.load(f)
    with torch.no_grad():
        content = await image.read()
        img = Image.open(io.BytesIO(content))
        index = predict(img)
        return {
            "debug_upload_filename": image.filename,
            "debug_name":  class_names[int(index)],
            "debug_index": str(index),
            "data":data[int(index)]
        }


# @app.get("/list")
# async def getsubtrat():
#     return [i for i in l1 if i in class_names]

# @app.get("/data")
# async def get_data():
#     with open("data.json") as f:
#       data = json.load(f)
#     l = []
#     for i in class_names:
#         for j in range(len(data)):
#             if i == data[j]["title"].lower():
#                 l.append(data[j])
#     return l

@app.get("/")
async def hello():
    return {"ping": "pong"}


# def prediction(text):
#     encoded = config.TOKENIZER.encode_plus(
#         text,
#         add_special_tokens=True,
#         max_length=config.MAX_LEN,
#         pad_to_max_length=True,
#         return_attention_mask=True,
#     )
#     ids = torch.tensor(encoded["input_ids"], dtype=torch.long).unsqueeze(0)
#     masks = torch.tensor(encoded["attention_mask"], dtype=torch.long).unsqueeze(0)
#     t_id = torch.tensor(encoded["token_type_ids"], dtype=torch.long).unsqueeze(0)
#     ids = ids.to(device, dtype=torch.long)
#     masks = masks.to(device, dtype=torch.long)
#     t_id = t_id.to(device, dtype=torch.long)
#     with torch.no_grad():
#         output = model(ids=ids, masks=masks, token_type_ids=t_id)
#         return torch.sigmoid(output).cpu().detach().numpy()


def predict(image_content):
    transformations = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image_tensor = transformations(image_content).float()
    # Add an extra batch dimension since pytorch treats all as batches
    image_tensor = image_tensor.unsqueeze_(0)

    if torch.cuda.is_available():
        image_tensor.cuda()

    input = Variable(image_tensor)
    output = model(input)
    pred = torch.argmax(output,1)
    print(pred)
    index = output.data.numpy().argmax()
    return index
