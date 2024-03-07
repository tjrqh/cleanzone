import json
from io import BytesIO
from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import resnet50
import base64
import requests
import shutil
import os
from flask_cors import CORS
from requests_toolbelt.multipart.encoder import MultipartEncoder


app = Flask(__name__)
CORS(app)


# 'device' 정의
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 'best_model' 경로 정의
best_model_path = 'best_model(992).pth'

# 사전 훈련된 ResNet50 모델 로드 (맨 위의 레이어는 제외)
model_base = resnet50(pretrained=True)
model_base = nn.Sequential(*list(model_base.children())[:-1])  # 마지막 FC 레이어 제외


# 모델 구성
class CustomResNet(nn.Module):
    def __init__(self, dropout_rate=0.5, dropconnect_rate=0.5):
        super(CustomResNet, self).__init__()
        self.model_base = nn.Sequential(*list(model_base.children())[:-1])
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(2048, 128)
        self.fc2 = nn.Linear(128, 256)
        self.dropout = nn.Dropout(dropout_rate)  # Dropout 레이어
        self.dropconnect1 = nn.Dropout2d(dropconnect_rate)  # DropConnect 레이어
        self.dropconnect2 = nn.Dropout2d(dropconnect_rate)  # DropConnect 레이어
        self.fc3 = nn.Linear(256, 4)  # 클래스 수에 맞게 조정

    def forward(self, x):
        x = self.model_base(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = nn.functional.softmax(x, dim=1)  # 활성화 함수 수정

        return x


# 모델과 클래스 인덱스 매핑 로드
best_model = CustomResNet(dropout_rate=0.5, dropconnect_rate=0.5).to(device)
best_model.eval()

class_indices = {0: 'normal', 1: 'mild', 2: 'moderate', 3: 'severe'}

# GPT-4 Vision API Key
gpt4_vision_api_key = "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"


# Function to encode the image
def encode_image(image_path_or_url):
    if image_path_or_url.startswith(('http://', 'https://')):
        return image_path_or_url
    else:
        with open(image_path_or_url, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


# Function to get GPT-4 Vision API prompt based on predicted class
def get_gpt4_prompt(predicted_class):
    prompts = {
        'normal': "The skin seems to be free of acne or any acne-related marks. Sebum seems balanced, maintaining a healthy skin condition. Search for possible signs of blackheads and whiteheads",
        'mild': "Some acne or pimples may be present, but their number is minimal. Small pimples are scattered on the skin surface. Occasional acne breakouts occur with relatively mild inflammation. Find signs of other skin diseases. Inspection from a professional is recommended.",
        'moderate': "Acne is noticeably developed, occurring in various areas of the face. Inflammatory acne is more prevalent, and some pimples may protrude or appear red on the skin surface. Find signs of other skin diseases. The subject should get treatment from a professional.",
        'severe': "Numerous acne lesions are present, accompanied by severe inflammation. Large pimples or nodules protrude on the skin surface, appearing red and inflamed. Besides acne, the skin may exhibit scarring, inflammation, and swelling. The skin tone may be irregular. Find signs of other skin diseases. The subject should get treatment immediately from a professional."
    }
    return prompts.get(predicted_class, "Default prompt for unknown class.")


# GPT-4 Vision API 호출 함수
def gpt4_vision_api(image_data, gpt4_prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {gpt4_vision_api_key}"
    }

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {"role": "system", "content": "You are a skincare advisor. Describe the skin picture. Answer only in korean. "},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": gpt4_prompt,
                        "max_tokens": 75,
                        "temperature": 0.7,
                        "top_p": 1
                    },

                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()


def extract_filename(file_path):
    return file_path.split("/")[-1]

# Flask API 엔드포인트
@app.route('/predict', methods=['POST'])
def predict_and_describe():
    header = request.headers.get('Authorization')
    try:
        # 받은 이미지를 저장할 임시 디렉토리
        temp_dir = 'temp_images/'

        # # 외부 서버에서 이미지 데이터 받음
        # image_data = request.files['file']

        # 전달받은 이미지 파일 저장
        image_paths = []
        for key, file in request.files.items():
            if file.filename:
                filename = f"{temp_dir}{key}.jpg"
                file.save(filename)
                image_paths.append(filename)

        # 이미지를 모델이 예측할 수 있는 형태로 변환
        transform = transforms.Compose([
            transforms.Resize(150),
            transforms.CenterCrop(150),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        predictions = []

        
        for img_path in image_paths:
            # image = Image.open(img_path)
            image = Image.open(img_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            # 모델 예측
            with torch.no_grad():
                output = best_model(image)
                _, predicted = torch.max(output.data, 1)
                predicted_class = class_indices[predicted.item()]

            # GPT-4 Vision API 호출을 위한 prompt 가져오기
            gpt4_prompt = get_gpt4_prompt(predicted_class)

            # GPT-4 Vision API 호출
            image_data = encode_image(img_path)
            gpt4_response = gpt4_vision_api(image_data, gpt4_prompt)

            # 결과 저장
            predictions.append(
                {'image_path': img_path, 'predicted_class': predicted_class, 'gpt4_response': gpt4_response})

        # 최종 결과 반환
        result = {'predictions': predictions}
        
        print(result)
        korean_contents = [entry["gpt4_response"]["choices"][0]["message"]["content"] for entry in result["predictions"]]
            #결과를 콘솔에 출력
        # for entry in predictions:
        #     print(f"Image Path: {entry['image_path']}")
        #     print(f"Predicted Class: {entry['predicted_class']}")
        #     print(f"GPT-4 Response: {entry['gpt4_response']['choices'][0]['message']['content']}")
        #     print("\n")

        # 파일 열기

        forehead_content = open('./temp_images/forehead.jpg', 'rb').read()
        left_cheek_content = open('./temp_images/leftCheek.jpg', 'rb').read()
        right_cheek_content = open('./temp_images/rightCheek.jpg', 'rb').read()
        chin_content = open('./temp_images/chin.jpg', 'rb').read()

        # 파일 이름 추출
        forehead_filename = extract_filename('./temp_images/forehead.jpg')
        left_cheek_filename = extract_filename('./temp_images/leftCheek.jpg')
        right_cheek_filename = extract_filename('./temp_images/rightCheek.jpg')
        chin_filename = extract_filename('./temp_images/chin.jpg')

        result_text = '\n'.join(korean_contents)


        multipart_data = MultipartEncoder(
            fields={
                'forehead': (forehead_filename, forehead_content),
                'leftCheek': (left_cheek_filename, left_cheek_content),
                'rightCheek': (right_cheek_filename, right_cheek_content),
                'chin': (chin_filename, chin_content),
                'AImessage':('AImessage', result_text)
            }
        )

        # 요청 헤더 설정
        headers = {'Content-Type': multipart_data.content_type, 'Authorization': header}

        # 파일을 서버로 업로드
        response_spring = requests.post("http://192.168.0.10:8761/face-picture/upload", headers=headers,data=multipart_data)
        # HTTP 상태 코드 확인
        if response_spring.status_code == 200:  # 성공적인 응답
            shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)  # 빈 폴더를 다시 생성
            print("요청이 성공적으로 처리되었습니다.")
            print("응답 내용:", response_spring.text)
        elif response_spring.status_code == 404:  # 404 Not Found 등 특정 상태 코드에 대한 처리
            print("서버에서 해당 리소스를 찾을 수 없습니다.")
        else:
            print("요청이 실패했습니다. 상태 코드:", response_spring.status_code)
            print("에러 메시지:", response_spring.text)

        return jsonify(response_spring.text)

    except Exception as e:
        return str(e)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)