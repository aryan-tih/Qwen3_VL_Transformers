from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
import base64


def image_to_url(image_path):
    with open(image_path,'rb') as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
    return base64_image



base64_image = image_to_url('')

model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-235B-A22B-Instruct", dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-235B-A22B-Instruct")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": f"data:image/jpeg;base64,{base64_image}",
            },
            {"type": "text", "text": "Convert the given image to HTML. Get data-bbox in it"},
        ],
    }
]

inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)

device = model.device
inputs = {k: v.to(device) for k,v in inputs.items()}

generated_ids = model.generate(**inputs, max_new_tokens=2048)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
]


output_text = processor.batch_decode(
    generated_ids_trimmed,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

print(output_text)
