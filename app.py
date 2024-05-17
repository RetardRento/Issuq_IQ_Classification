import tensorflow as tf
from transformers import AutoTokenizer, TFBertModel
from fastapi import FastAPI, Request
import numpy
import re
from tensorflow import keras

with keras.utils.custom_object_scope({"TFBertModel": TFBertModel}):
    model = tf.keras.models.load_model("model_reqs/better_model.h5")


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

encoded_dict = {
    "AUDIO": 0,
    "BLOATWARE": 1,
    "OPERATING SYSTEM": 2,
    "CAMERA": 3,
    "CONNECTIVITY": 4,
    "PHYSICAL DAMAGE": 5,
    "SOFTWARE UPDATES": 6,
    "BATTERY": 7,
    "DISPLAY": 8,
    "APP ISSUES": 9,
}


app = FastAPI()


@app.post("/classify")
async def classify_text(request: Request):
    json_obj = await request.json()

    text = json_obj.get("text")

    x_val = tokenizer(
        text=text,
        add_special_tokens=True,
        max_length=12,
        truncation=True,
        padding="max_length",
        return_tensors="tf",
        return_token_type_ids=False,
        return_attention_mask=True,
        verbose=True,
    )

    input_ids = x_val["input_ids"].numpy()
    attention_mask = x_val["attention_mask"].numpy()

    # making prediction
    validation = (
        model.predict({"input_ids": input_ids, "attention_mask": attention_mask}) * 100
    )
    validation = validation.tolist()  # convert numpy.float32 to list

    top_2_cat = sorted(
        sorted(
            zip(encoded_dict.keys(), validation[0]), key=lambda x: x[1], reverse=True
        )[:2]
    )

    # return {
    #     category: value
    #     for category, value in top_2_cat
    #     if not (category == "camera" and value <= float(92))
    # }

    result = {category: value for category, value in top_2_cat}

    pre_defined = [
        "Audio",
        "Bloatware",
        "Operating System",
        "Camera",
        "Connectivity",
        "Physical Damage",
        "Software Updates",
        "Battery",
        "Display",
        "App Issues",
        "Connect",
        "App",
        "WIFI",
        "NETWORK",
        "Wi-Fi",
    ]

    def find_matching_words(text, word_list):
        # Convert text and word_list to upper case
        text_upper = text.upper()
        word_list_upper = [word.upper() for word in word_list]

        # Initialize an empty list to store matching words
        matching_words = []

        # Iterate over each word in the word_list
        for word in word_list_upper:
            # Check if the word is in the text
            if word in text_upper:
                # If it matches, add the word to the matching_words list
                matching_words.append(word)

        # Return the list of matching words
        return matching_words

    pre_defined = [word.upper() for word in pre_defined]
    word_match = find_matching_words(text=text.upper(), word_list=pre_defined)
    set(word_match)

    def extract_categories(json_data):
        return set(json_data.keys())

    res = extract_categories(result)
    uni_res_word = set(word_match).union(res)

    return {"tags": uni_res_word, "weights": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
