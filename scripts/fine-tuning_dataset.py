import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from transformers import AutoProcessor, AutoTokenizer, AutoModel, AutoModelForCausalLM, LlavaNextForConditionalGeneration, AutoModelForZeroShotObjectDetection
from PIL import Image, ImageDraw, ImageColor, ImageFont
from llama_cpp import Llama
from tqdm import tqdm
import numpy as np
import torch
import copy
import json
import ast
import re


device = 'cuda:0'
image_to_text_model = 'llava-hf/llava-v1.6-mistral-7b-hf'
segmentation_model = "IDEA-Research/grounding-dino-base"
embeddings_model = 'Alibaba-NLP/gte-large-en-v1.5'

def load_models():
    processor_itt = AutoProcessor.from_pretrained(image_to_text_model)
    model_itt = LlavaNextForConditionalGeneration.from_pretrained(image_to_text_model,
                                                low_cpu_mem_usage = True,
                                                torch_dtype=torch.float16,
                                                device_map=device,
                                                cache_dir='/mnt/esperanto/et/huggingface/hub',
                                                )

    processor_segmentation = AutoProcessor.from_pretrained(segmentation_model)
    model_segmentation = AutoModelForZeroShotObjectDetection.from_pretrained(segmentation_model,
                                                                    low_cpu_mem_usage = True,
                                                                    device_map=device,
                                                                    cache_dir='/mnt/esperanto/et/huggingface/hub',
                                                                    )

    tokenizer_embeddings = AutoTokenizer.from_pretrained(embeddings_model)
    model_embeddings = AutoModel.from_pretrained(embeddings_model,
                                        low_cpu_mem_usage = True,
                                        torch_dtype=torch.float16,
                                        device_map=device,
                                        trust_remote_code=True,
                                        cache_dir='/mnt/esperanto/et/huggingface/hub',
                                        )
                        
    return model_itt, processor_itt, model_segmentation, processor_segmentation, model_embeddings, tokenizer_embeddings

def get_labels_color(labels):
    all_colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'pink', 'brown', 'gray', 'cyan', 'magenta', 'lime', 'teal', 'navy', 'maroon', 'olive', 'indigo', 'violet', 'coral', 'salmon', 'gold', 'silver', 'turquoise', 'lavender', 'beige', 'tan', 'mint', 'plum', 'khaki', 'ivory', 'honeydew']
    labels = list(set(labels))

    labels_to_colors = [[label, color] for label, color in zip(labels, all_colors)]

    return dict(labels_to_colors)

def save_image_with_boxes(image, segments, path='./output_image.jpg'):
    boxes, labels, scores = segments[0]['boxes'], segments[0]['labels'], segments[0]['scores']
    
    labels_to_colors = get_labels_color(labels)

    img = image.copy().convert("RGBA")

    for box, label, score in zip(boxes, labels, scores):

        box_width = max(1, int(0.0035 * max(img.size)))
        text_size = max(9, int(0.01 * max(img.size)))


        overlay = Image.new('RGBA', img.size, (0,0,0,0))
        draw = ImageDraw.Draw(overlay)

        transparency = 200
        box = box.cpu().tolist()
        outline_color = ImageColor.getcolor(labels_to_colors[label], "RGB")
        draw.rectangle(box, outline=(outline_color[0], outline_color[1], outline_color[2], transparency), width=box_width)

        font = ImageFont.load_default()
        font = font.font_variant(size=text_size)

        position = (box[0] + box_width, box[1] + box_width)
        text_bbox = draw.textbbox(position, label, font=font)
        text_width = text_bbox[2] - text_bbox[0] + 2
        text_height = text_bbox[3] - text_bbox[1] + 2
        background_rectangle = [position[0], position[1], position[0] + text_width, position[1] + text_height]
        background_rectangle = [val + 2 if i%2==1 else val for i, val in enumerate(background_rectangle)] #add some margin to fit the text position

        draw.rectangle(background_rectangle, fill='grey')
        draw.text(position, label, fill='white', font=font)

        img = Image.alpha_composite(img, overlay)

    img = img.convert("RGB")
    img.save(path)
    return path

def get_labels(model, processor, image, n_objects=10, n_parallel_inference=3):
    generation_success = False
    while not generation_success:
        outputs = controled_generation(model, processor, [image] * n_parallel_inference, n_objects=n_objects, do_sample=True, temperature=0.7)
        try:
            list_outputs = []
            for output in outputs:
                list_outputs.append(ast.literal_eval(output))
            # if len(list_outputs[-1]['objects']) <= n_objects*1.2 and len(list_outputs[-1]['objects']) >= n_objects*0.8:
            generation_success = True
            # else:
            #     print(f"Failed to generate the right number of labels: {len(list_outputs[-1]['objects'])}")

        except:
            print(f"An error occured during json evaluation of generated output: {outputs}")
            continue
    
    list_labels = ['. '.join(list_outputs[i]['objects']).lower() +'.' for i in range(n_parallel_inference)]
    #### check if a list of titles can be usefull
    title = list_outputs[0]['title']

    return list_labels, title
    
def run_segmentation(model, processor, image, list_labels):
    n_parallel = len(list_labels)
    inputs = processor(images=[image] * n_parallel, text=list_labels, return_tensors="pt", padding=True).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(outputs,
                                                            inputs.input_ids,
                                                            box_threshold=0.2,
                                                            text_threshold=0.2,
                                                            target_sizes=[image.size[::-1]] * n_parallel
                                                            )

    results = [{
                'scores': torch.concatenate([results[i]['scores'] for i in range(n_parallel)]),
                'labels': [l for sublist in [results[i]['labels'] for i in range(n_parallel)] for l in sublist],
                'boxes': torch.concatenate([results[i]['boxes'] for i in range(n_parallel)])
            }]

    return results

def compute_overlapping_area(boxes):
    areas = np.zeros((boxes.shape[0], boxes.shape[0]))

    for i in range(boxes.shape[0]):
        for j in range(boxes.shape[0]):
            x0 = max(boxes[i, 0], boxes[j, 0])
            y0 = max(boxes[i, 1], boxes[j, 1])
            x1 = min(boxes[i, 2], boxes[j, 2])
            y1 = min(boxes[i, 3], boxes[j, 3])

            width = max(0, x1 - x0)
            height = max(0, y1 - y0)

            base_box_area = max((boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]), 1e-5)
            intersection_area = min(width * height / base_box_area, 1)

            areas[i,j] = intersection_area
    
    return areas

def get_overlapping_index(intersection_areas, current_label, segments, threshold=.7):
    indices_to_remove = []
    real_indices = [k for k, val in enumerate(segments[0]['labels']) if val==current_label]

    for i in range(intersection_areas.shape[0]):
        for j in range(intersection_areas.shape[1]):
            if i == j or real_indices[i] in indices_to_remove or real_indices[j] in indices_to_remove:
                continue
            if intersection_areas[i,j] >= threshold:
                indices_to_remove.append(real_indices[i])

    return indices_to_remove

def remove_boxes(segments, boxes_to_remove):
    n = segments[0]['scores'].shape[0]
    new_scores = torch.tensor([segments[0]['scores'].tolist()[i] for i in range(n) if i not in boxes_to_remove], device=device)
    new_labels = [segments[0]['labels'][i] for i in range(n) if i not in boxes_to_remove]
    new_boxes = torch.tensor([segments[0]['boxes'].tolist()[i] for i in range(n) if i not in boxes_to_remove], device=device)
    
    return [{'scores': new_scores, 'labels': new_labels, 'boxes': new_boxes}]

def get_overlapping_boxes(segments):  #overlapping w/ same labels
    unique_labels = list(set(segments[0]['labels']))
    boxes_to_remove = []

    for l in unique_labels:
        indices = [i for i in range(len(segments[0]['labels'])) if segments[0]['labels'][i] == l]
        boxes = np.array([segments[0]['boxes'][i].cpu() for i in indices])
        intersection_areas = compute_overlapping_area(boxes)
        boxes_to_remove.extend(get_overlapping_index(intersection_areas, l, segments))

    boxes_to_remove = list(set(boxes_to_remove))
    return boxes_to_remove

def get_similar_boxes(segments, threshold=.75): #similar w/ different (or same) labels
    boxes_to_remove = []

    intersection_areas = compute_overlapping_area(segments[0]['boxes'].cpu())

    for i in range(intersection_areas.shape[0]):
        for j in range(i):
            if intersection_areas[i,j] >= threshold and intersection_areas[j,i] >= threshold:
                idx = i if segments[0]['scores'][i] < segments[0]['scores'][j] else j
                boxes_to_remove.append(idx)

    boxes_to_remove = list(set(boxes_to_remove))
    return boxes_to_remove

def merge_labels(model, tokenizer, segments, display_similarity=False, threshold=.85):
    real_labels = list(set(segments[0]['labels']))
    inputs = tokenizer(real_labels, max_length=8192, padding=True, truncation=True, return_tensors='pt').to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)

    embeddings = outputs.last_hidden_state[:, 0]
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    scores = (embeddings @ embeddings.T)

    similar_pairs = {}
    for i in range(scores.shape[0]):
        for j in range(i):
            if display_similarity:
                print(f"similarity({real_labels[i], real_labels[j]})={scores[i,j]}")
            if scores[i,j] >= threshold:
                label_to_be_replaced = real_labels[i] if len(real_labels[i]) >= len(real_labels[j]) else real_labels[j]
                label_to_replace_with = real_labels[i] if label_to_be_replaced == real_labels[j] else real_labels[j]
                similar_pairs[label_to_be_replaced] = label_to_replace_with

    all_merged = False
    while not all_merged:
        all_merged = True
        for k in range(len(segments[0]['labels'])):
            if segments[0]['labels'][k] in similar_pairs.keys():
                #print(f"Merging label '{segments[0]['labels'][k]}' into '{similar_pairs[segments[0]['labels'][k]]}'...")
                segments[0]['labels'][k] = similar_pairs[segments[0]['labels'][k]]
                all_merged = False

    return segments
    
def controled_generation(model, processor, image, n_objects=10, **kwargs):
    n_parallel = len(image)
    prompt = f"""[INST] <image>\nAnalyze the scene and infer what it is representing. Given the scene, list 5 to 10 objects or entities most likely to be part of the scene and most important to spot (use ONE SIMPLE WORD ONLY to describe an object or entity - this will be reprenting a category in large meaning). Ignore objects that are small or irrelevant to a blind person. Answer by filling out the following JSON format. Your answer must be parse-able with python's ast.literal_eval() - DO NOT ADD ANYTHING ELSE:
    {{
        "title": "short_scene_title",
        "objects": {[f"object_{i}" for i in range(n_objects)]}
    }}
    [/INST]
    """

    starter = """   {
        "title": \""""

    inputs = processor([prompt + starter] * n_parallel, image, return_tensors='pt').to(device, torch.float16)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=50, tokenizer=processor.tokenizer, stop_strings='\",', **kwargs)

    intermediary = """
        \"objects\": ['"""

    outputs = processor.tokenizer.batch_decode(outputs, skip_special_tokens=False)
    outputs = [re.sub(r'<s>|</s>|<pad>', '', o) + "\n    \"objects\": ['" for o in outputs]

    inputs['input_ids'], inputs['attention_mask'] = processor.tokenizer(outputs, return_tensors='pt', padding=True, add_special_tokens=False).to(device).values()

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=300, tokenizer=processor.tokenizer, stop_strings=["]", "dummy string to circumvent bug in stop_strings"], **kwargs)

    ending = """
        }"""

    outputs = processor.tokenizer.batch_decode(outputs, skip_special_tokens=False)
    outputs = [re.sub(r'<s>|</s>|<pad>', '', o)[len(prompt):] + "\n    }" for o in outputs]
    
    return outputs
    
def post_processing(model_embeddings, tokenizer_embeddings, segments):
    ## merging labels that are close in semantics
    segments_merged_labels = merge_labels(model_embeddings, tokenizer_embeddings, segments, display_similarity=False)

    ## removing boxes that are overlapping a lot with another same-label box
    boxes_to_remove = get_overlapping_boxes(segments_merged_labels)
    segments_sparser_boxes = remove_boxes(segments_merged_labels, boxes_to_remove)

    ## removing boxes that are almost the same as an other but with a different label (keeping the one with highest score)
    boxes_to_remove = get_similar_boxes(segments_sparser_boxes)
    segments_cleaned = remove_boxes(segments_sparser_boxes, boxes_to_remove)

    return segments_cleaned

def run_image_description(model_llm, image, title, segments):
    system_prompt = "You are an AI model designed to help visually impaired people. Your task is to provide a comprehensive description of the image, locating important objects to guide disabled people through the scene."

    width, height = image.size
    segments[0]['boxes'][:, 0::2] /= width
    segments[0]['boxes'][:, 1::2] /= height

    boxes = [row for row in segments[0]['boxes'].tolist()]
    boxes = [str([round(x,2) for x in row]) for row in boxes]

    labels = segments[0]['labels']

    box_prompt = '\n'.join(sorted([a + ' ' + b for a,b in zip(labels, boxes)]))

    question_prompt = f"Below is a description of a {title} scene, along with a list of objects present in the scene along with their coordinates following the format 'object [x_min, y_min, x_max, y_max]'. Provide a descriptive paragraph using a human-like description, do not mention coordinates. Only use the position information and infer from it, do not add any comment or guess. Remain factual and avoid unnecessary embellishments, keep it simple."

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

    {question_prompt}
    {box_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    """

    outputs = model_llm(
       prompt,
       max_tokens=500,
       echo=False,
   )

    return outputs["choices"][0]["text"].strip()



def main():
    model_itt, processor_itt, model_segmentation, processor_segmentation, model_embeddings, tokenizer_embeddings = load_models()

    model_llm = Llama(model_path="../Models_Tests/Meta-Llama-3-70B-Instruct.Q4_K_M.gguf",
            n_ctx=4096,
            n_threads=8,            # The number of CPU threads to use, tailor to your system and the resulting performance
            n_gpu_layers=-1,         # The number of layers to offload to GPU, if you have GPU acceleration available, if -1 all layers are offloaded
            verbose=False,
            )
    

    data_dir = 'data/val2014/'
    file_list = os.listdir(data_dir)

    outputs = []

    for i, filename in tqdm(enumerate(file_list)):

        image = Image.open(data_dir + filename).convert('RGB')

        list_labels, title = get_labels(model_itt, processor_itt, image, n_objects=7, n_parallel_inference=3)

        segments = run_segmentation(model_segmentation, processor_segmentation, image, list_labels)

        segments_postprocessed = post_processing(model_embeddings, tokenizer_embeddings, segments)

        output_path = save_image_with_boxes(image, segments_postprocessed, data_dir + 'boxed/' + filename[:-4] + '_boxed.jpg')

        descriptive_text = run_image_description(model_llm, image, title, segments_postprocessed)

        segments[0]['boxes'] = segments[0]['boxes'].tolist()
        segments[0]['scores'] = segments[0]['scores'].tolist()
        segments_postprocessed[0]['boxes'] = segments_postprocessed[0]['boxes'].tolist()
        segments_postprocessed[0]['scores'] = segments_postprocessed[0]['scores'].tolist()
        outputs.append({
            'idx': i,
            'image_path': data_dir + filename,
            'boxed_image_path': data_dir + 'boxed/' + filename[:-4] + '_boxed.jpg',
            'title': title,
            'original_segments': segments,
            'segments_postprocessed': segments_postprocessed,
            'generated_descriptive_text': descriptive_text,
        })

        with open(f'###TBD{data_dir[:-1]}_descriptive_texts.json', 'w') as file:
            json.dump(outputs, file, indent=4)




if __name__ == '__main__':
    main()

        