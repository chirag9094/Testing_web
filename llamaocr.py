import together
import re
import os
from dotenv import load_dotenv
from together import Together
import base64
from PIL import Image, ImageDraw
from IPython.display import Markdown, display
from matplotlib import pyplot as plt
from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from termcolor import colored

os.environ["TOGETHER_API_KEY"] = '10c1c7c6fabe12373eee8e5ef785d62396cb7e35c0500e6eab8097f3c5fd2187' 
client = Together()

load_dotenv()
ai_endpoint = os.getenv('AI_SERVICE_ENDPOINT')
ai_key = os.getenv('AI_SERVICE_KEY')

if not ai_endpoint or not ai_key:
     raise ValueError("AI_SERVICE_ENDPOINT or AI_SERVICE_KEY is not set in the .env file.")

cv_client = ImageAnalysisClient(
            endpoint=ai_endpoint,
            credential=AzureKeyCredential(ai_key))

getDescriptionPrompt = "Detect the text present in the image and no extra information. Also no markup require. Display the text in simple string format"
imagePath = r"C:\Users\Chirag C\OneDrive\Desktop\test.jpg"

def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
lla_arr=[]
print("\nLLAMA")
try:
    base64_image = encode_image(imagePath)
    stream = client.chat.completions.create(
        model="meta-llama/Llama-Vision-Free",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": getDescriptionPrompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
        stream=True,
    )

    # Improved stream handling with proper checks
    for chunk in stream:
        # Check if chunk has choices
        if not hasattr(chunk, 'choices') or not chunk.choices:
            continue

        # Check if first choice exists and has delta
        choice = chunk.choices[0]
        if not hasattr(choice, 'delta'):
            continue

        # Check if delta has content
        delta = choice.delta
        if not hasattr(delta, 'content'):
            continue

        # Print content if it exists
        content = delta.content
        if content is not None:
            lla_arr.append(content)
            print(content, end="", flush=True)

except FileNotFoundError:
    print(f"Error: Could not find image file at {imagePath}")
except Exception as e:
    print(f"An error occurred: {str(e)}")


#Azure_AI
print("\n\nAzure")
with open(imagePath, 'rb') as f:
        image_data = f.read()
# Call Azure Vision API
res = cv_client.analyze(
        image_data=image_data,
        visual_features=[VisualFeatures.READ]   )

if not res.read or not res.read.blocks:
    print("No text found in the image.")
    

# print("\nDetected Text:")
image = Image.open(imagePath)
fig = plt.figure(figsize=(image.width / 100, image.height / 100))
plt.axis('off')
draw = ImageDraw.Draw(image)
color = 'cyan'
all_lines = []
for block in res.read.blocks:
    for line in block.lines:
        all_lines.append(line.text)
        print(line.text)
        r = line.bounding_polygon
        bounding_polygon = [(r[0].x, r[0].y), (r[1].x, r[1].y), (r[2].x, r[2].y), (r[3].x, r[3].y)]
        draw.polygon(bounding_polygon, outline=color, width=3)


# Text Conversion
raw_text_l = ''.join(lla_arr)
clean_text_l = ' '.join(raw_text_l.split())
clean_tex_l = clean_text_l[:-1]
raw_text_a = ' '.join(all_lines)

def normalize(text):
    return ''.join(text.lower().replace('\n', ' ').split())

if normalize(clean_text_l) == normalize(raw_text_a):
    print("\nThe texts are the same (ignoring case and spaces).")
else:
    print("\nThe texts are different.")

# # Text Comparison
# def normalize11(text):
#     text = text.lower()
#     text = re.sub(r'[^\w@-]', ' ', text)  # Keep words, hyphens, @ for emails
#     words = text.split()
#     return words

# words1 = set(normalize11(clean_text_l))
# words2 = set(normalize11(raw_text_a))

def clean_word(word):
    return re.sub(r'[^\w@-]', '', word).lower()

# Normalize full text to list of cleaned words
words1 = set(clean_word(w) for w in clean_text_l.split())
words2 = set(clean_word(w) for w in raw_text_a.split())

# print(words1)
# print(words2)
# Common and unique words
common_words = words1 & words2
only_in_text2 = words2 - words1

print("Comparison (based on Text 2):\n")
for word in raw_text_a.split():
    normalized_word = re.sub(r'[^\w@-]', '', word).lower()
    if normalized_word in common_words:
        print(colored(word, "green"), end=' ')
    elif normalized_word in only_in_text2:
        print(colored(word, "red"), end=' ')
    else:
        print(word, end=' ')





# from termcolor import colored

# # Texts to compare
# text1 = "Roberto Tamburello Engineering Manager 555-123-4567 roberto@adventure-works.com"

# text2 = "Adventure Works Cycles Roberto Tamburello Engineering Manager roberto@adventure-works.com 555-123-4567"

# # Normalize function: lowercase + remove extra spaces
# def normalize(text):
#     return text.lower().split()

# print(text1)
# print(text2)

# # Get normalized word sets
# words1 = set(normalize(text1))
# words2 = set(normalize(text2))

# print(words1)
# print(words2)

# # Common and unique words
# common_words = words1 & words2
# only_in_text2 = words2 - words1

# # Output color-coded result based on Text 2
# print("Final Comparison Result:\n")
# for word in text2.replace('\n', ' ').split():
#     normalized_word = word.lower()
#     if normalized_word in common_words:
#         print(colored(word, 'green'), end=' ')
#     elif normalized_word in only_in_text2:
#         print(colored(word, 'blue'), end=' ')
#     else:
#         print(word, end=' ')