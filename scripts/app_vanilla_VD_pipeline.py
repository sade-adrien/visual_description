import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

import streamlit as st
from transformers import AutoProcessor, AutoTokenizer, AutoModel, AutoModelForCausalLM, LlavaNextForConditionalGeneration, AutoModelForZeroShotObjectDetection
from PIL import Image, ImageDraw, ImageColor, ImageFont
import numpy as np
import torch
import ast
import re

device = 'cuda:0'
image_to_text_model = 'llava-hf/llava-v1.6-mistral-7b-hf'
segmentation_model = "IDEA-Research/grounding-dino-base"
embeddings_model = 'Alibaba-NLP/gte-large-en-v1.5'

#def fcts
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

def save_image_with_boxes(image, segments):
    path = './output_image.jpg'

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
            if len(list_outputs[-1]['objects']) <= n_objects*1.2 and len(list_outputs[-1]['objects']) >= n_objects*0.8:
                generation_success = True
            else:
                print(f"Failed to generate the right number of labels: {len(list_outputs[-1]['objects'])}")
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
                print(f"Merging label '{segments[0]['labels'][k]}' into '{similar_pairs[segments[0]['labels'][k]]}'...")
                segments[0]['labels'][k] = similar_pairs[segments[0]['labels'][k]]
                all_merged = False

    return segments
    
def controled_generation(model, processor, image, n_objects=10, **kwargs):
    n_parallel = len(image)
    prompt = f"""[INST] <image>\nAnalyze the scene and infer what it is representing. Given the scene, list the {n_objects} objects or entities most likely to be part of the scene and most important to spot (use ONE SIMPLE WORD ONLY to describe an object or entity - this will be reprenting a category in large meaning). Ignore objects that are small or irrelevant to a blind person. Answer by filling out the following JSON format. Your answer must be parse-able with python's ast.literal_eval() - DO NOT ADD ANYTHING ELSE:
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


def show_page1():
    st.set_page_config(layout="wide")
    st.title('Visual Description / Automatic Annotation')
    col1, col2 = st.columns(2)
    output_path = None

    if 'models_loaded' not in st.session_state:
        with st.spinner('Loading models...'):
            model_itt, processor_itt, model_segmentation, processor_segmentation, model_embeddings, tokenizer_embeddings = load_models()
            st.session_state['models_loaded'] = True

            st.session_state['model_itt'] = model_itt
            st.session_state['processor_itt'] = processor_itt
            st.session_state['model_segmentation'] = model_segmentation
            st.session_state['processor_segmentation'] = processor_segmentation
            st.session_state['model_embeddings'] = model_embeddings
            st.session_state['tokenizer_embeddings'] = tokenizer_embeddings

    else:
        model_itt = st.session_state['model_itt']
        processor_itt = st.session_state['processor_itt']
        model_segmentation = st.session_state['model_segmentation']
        processor_segmentation = st.session_state['processor_segmentation']
        model_embeddings = st.session_state['model_embeddings']
        tokenizer_embeddings = st.session_state['tokenizer_embeddings']

    with col1:
        image_file = st.file_uploader("Upload Scene Image", type=["png","jpg","jpeg"])
        if image_file is not None:
            image = Image.open(image_file).convert('RGB')
            st.image(image, caption='', use_column_width=True)

        segment_button = st.button("Segment")
    

    if segment_button and (image_file is not None):
        list_labels, title = get_labels(model_itt, processor_itt, image, n_objects=10, n_parallel_inference=3)

        segments = run_segmentation(model_segmentation, processor_segmentation, image, list_labels)

        with col2:
            st.markdown("<br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
            st.write(list_labels)

        segments_postprocessed = post_processing(model_embeddings, tokenizer_embeddings, segments)

        output_path = save_image_with_boxes(image, segments_postprocessed)
    
    
    if output_path is not None:
        output_image = Image.open(output_path).convert('RGB')
        st.image(output_image, caption=title, use_column_width=True)




show_page1()
    
        

    