from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
import io
import logging
# new package for cron task
import aiocron
import aiohttp
from datetime import datetime, time, timedelta
# new package for recognize category and 
import re
import numpy as np
import pandas as pd
from rapidfuzz import process

app = FastAPI()

''' 
async def is_active_hours():
    now = datetime.now().time()
    return time(9,0) <= now <= time(18,0) # assume active time between 9:00 and 18:00
'''
@aiocron.crontab('*/10 * * * *')
async def self_ping():
    async with aiohttp.ClientSession() as session:
        async with session.get('https://iteration2-receiptapi.onrender.com/health') as response:
            print(f"Health check response: {response.status}")
   
@app.on_event("startup")
async def startup_event():
    self_ping.start()


# set log
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mindee_endpoint = "https://api.mindee.net/v1/products/mindee/expense_receipts/v5/predict"
mindee_api_key = "413c3140cb93daff17536ec583083000"

def get_item(item):
    item_description = item.get('description', '')
    item_cost = item.get('total_amount', 0)
    newItem = {
        'description':item_description,
        'cost': item_cost
    }
    return newItem

# the function required to get category of item
def get_category(Categories, group):
    for key,val in Categories.items():  
        if group in val:
            return key
    return "Other"

def load_food_data():
    return pd.read_csv('FoodItemsList.csv')

# the function to get expirydate
def get_expiry_date(food_name, food_group):
    food_data = load_food_data()
    best_match = process.extractOne(food_name, food_data['Name'].tolist())
    if best_match:
        matched_str, score, index = best_match
        food_item = food_data[food_data['Name'] == matched_str]
        if not food_item.empty:
            days_until_expiry = int(food_item['Days Until Expiry'].values[0])
            first_time = datetime.now() + timedelta(days=days_until_expiry)
            return first_time.strftime("%Y-%m-%d")
        else:
            second_match = process.extractOne(food_group, food['Name'].tolist())
            if second_match:
                second_food_item = food_data[food_data['Name'] == second_match[0]]
                if not second_food_item:
                    days2 = int(second_food_item['Days Until Expiry'].value[0])
                    second_time = datetime.now() + timedelta(days=days2)
                    return second_time.strftime("%Y-%m-%d")
    return None    

#get quantity info
def get_quantity(input_text):
    input_quantity = input_text.split(" ")[-1]
    if input_quantity == "":
        return "", ""
    elif input_quantity[0] in ['0','1','2','3','4','5','6','7','8','9']: #if the last text is quantity - get value and untis
        return(re.sub("[a-zA-Z//]*", "", input_quantity), (re.sub("[0-9\\.]*", "", input_quantity)).lower())
    elif input_quantity == "pack":
        return(input_text.split(" ")[-2], input_text.split(" ")[-1])
    else: #if the last text is not a quantity return empty 
        return "", ""
    

# Remove any brand name from the text
def get_info(pattern, Categories, product_groups, input_text):
    input_text = input_text.lower()
   
    #remove quantity
    #"[0-9]*pk|[0-9]*mg|[0-9]*gms|[0-9]*kg|[0-9]*ml|[0-9]*l"
    quantityValue, unit = get_quantity(input_text)
    quantity = quantityValue + unit
    input_text = re.sub(quantity, "", input_text)
    print("quantity:", quantity)
   
    #remove brand
    # Create a regex pattern to match the brands (case insensitive)
    cleaned_text = re.sub(pattern, '', input_text).strip()
    print("cleaned_text:", cleaned_text)
   
    #extract category
    #best_match = process.extractOne(cleaned_text, product_groups)
    #print("Category:", get_category(best_match[0]) )
     # Direct regex match for product groups
    best_match = ""
 
    for group in product_groups:
        if re.search(r'\b' + group + r'\b', cleaned_text):
            best_match = group
            break
   
    #return(f"Best match product group: {best_match[0]} with a confidence of {best_match[1]}")
    #print(f"product group: {best_match[0]}")
    return {"item": cleaned_text.title(), "category": get_category(Categories, best_match), "quantity": quantity, "quantityValue":quantityValue, "unit": unit, "expirydate": get_expiry_date(cleaned_text, best_match), "source": 'scan', "product_group": best_match}
    #"product group": best_match[0]


def assemble_info(pattern, Categories, product_groups, original_list):
    item_list = []
    for i in range(len(original_list)):
        try:
            processed_item = get_info(pattern, Categories, product_groups, original_list[i].get('description',''))
            processed_item['cost'] = original_list[i]['cost']
            item_list.append(processed_item)
        except Exception as e:
            print(f"Error when handling item: {str(e)}. Skip this item.")
    return item_list


@app.post("/upload/")
async def create_upload_file(uploaded_file: UploadFile = File(...)):
    if not uploaded_file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
        raise HTTPException(status_code=415, detail="Unsupported file type")
    
    try:
        contents = await uploaded_file.read()
        logger.info(f"File size: {len(contents)} bytes")
        #使用Mindee API
        headers = {
            "Authorization": f"Token {mindee_api_key}"
        }
        files = {
            "document": (uploaded_file.filename, contents, uploaded_file.content_type)
        }
        r = requests.post(mindee_endpoint, headers=headers, files=files)
        logger.info(f"OCR Service Response status code: {r.status_code}")
        logger.info(f"OCR Service Response content: {r.text[:200]}...")  # 记录响应的前200个字符

        r.raise_for_status()  # 如果响应状态码不是200，将引发异常
        data = r.json() 

        #extract the proper item name, quantity and category
        #files needed to extract values are below;
        #required files
        with open('Categories.json', 'r') as file:
            Categories = json.load(file)
        product_groups = list(np.load("product_groups.npy", allow_pickle = True))

        with open("brand_names.txt", "r") as file:
            pattern = file.read()
        

        if 'document' in data and 'inference' in data['document'] and 'prediction' in data['document']['inference']:
            line_items = data['document']['inference']['prediction'].get('line_items', [])
            #get item names
            
            intermediate_list = [get_item(item) for item in line_items ]
            item_list = assemble_info(pattern, Categories, product_groups, intermediate_list)
        return {"item_list": item_list}
        
    except requests.RequestException as e:
        logger.error(f"OCR Service request fail: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR Service Request fail: {str(e)}")
    except json.JSONDecodeError as e:
        logger.error(f"OCR Service Returns Useless JSON: {str(e)}")
        raise HTTPException(status_code=500, detail="OCR Service Returns Useless JSON")
    except Exception as e:
        logger.error(f"File Processing Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"File Processing Error: {str(e)}")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}
