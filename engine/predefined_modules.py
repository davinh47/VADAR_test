import re
import torch
import numpy as np
import io
import json
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    GenerationConfig,
)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from unidepth.models import UniDepthV2
import groundingdino.datasets.transforms as T

from VADAR_test.prompts.vqa_prompt import (
    VQA_PROMPT_CLEVR,
    VQA_PROMPT_GQA,
    VQA_PROMPT_GQA_HOLISTIC,
    VQA_PROMPT,
)
from .engine_utils import *
from groundingdino.util.inference import load_model, predict


class PredefinedModule:
    def __init__(self, name, trace_path=None):
        self.trace_path = trace_path
        self.name = name

    def write_trace(self, html):
        if self.trace_path:
            with open(self.trace_path, "a+") as f:
                f.write(f"{html}\n")


class OracleModule(PredefinedModule):
    def __init__(self, name, trace_path=None):
        super().__init__(name, trace_path)
        self.reference_image = None
        self.scene_json = None
        self.oracle = None

    def set_oracle(self, oracle, reference_image, scene_json):
        self.oracle = oracle
        self.reference_image = reference_image
        self.scene_json = scene_json

    def clear_oracle(self):
        self.reference_image = None
        self.scene_json = None
        self.oracle = None


class LocateModule(OracleModule):
    def __init__(
        self,
        dataset,
        grounding_dino=None,
        molmo_processor=None,
        molmo_model=None,
        trace_path=None,
    ):
        super().__init__("loc", trace_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = dataset

        if self.dataset in ["clevr", "gqa"]:
            self.molmo_processor = molmo_processor
            self.molmo_model = molmo_model
        else:
            self.grounding_dino = grounding_dino
            self.BOX_THRESHOLD = 0.25
            self.TEXT_TRESHOLD = 0.25

    def _extract_points(self, molmo_output, image_w, image_h):
        all_points = []
        for match in re.finditer(
            r'x\d*="\s*([0-9]+(?:\.[0-9]+)?)"\s+y\d*="\s*([0-9]+(?:\.[0-9]+)?)"',
            molmo_output,
        ):
            try:
                point = [float(match.group(i)) for i in range(1, 3)]
            except ValueError:
                pass
            else:
                point = np.array(point)
                if np.max(point) > 100:
                    # Treat as an invalid output
                    continue
                point /= 100.0
                x = int(point[0] * image_w)
                y = int(point[1] * image_h)
                all_points.append([x, y])

        # convert all points to int
        return all_points

    def _parse_bounding_boxes(self, boxes, width, height):
        if len(boxes) == 0:
            return []

        bboxes = []
        for box in boxes:
            cx, cy, w, h = box
            x1 = cx - 0.5 * w
            y1 = cy - 0.5 * h
            x2 = cx + 0.5 * w
            y2 = cy + 0.5 * h
            bboxes.append(
                [
                    int(x1 * width),
                    int(y1 * height),
                    int(x2 * width),
                    int(y2 * height),
                ]
            )
        return bboxes

    def transform_image(self, og_image):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        og_image = og_image.convert("RGB")
        img = np.asarray(og_image)
        im_t, _ = transform(og_image, None)
        return img, im_t

    def execute_pts(self, image, object_prompt):
        original_object_prompt = object_prompt
        if self.oracle:
            pts = self.oracle.locate(image, object_prompt, self.scene_json)
        else:
            if object_prompt[-1] != "s":
                object_prompt = object_prompt + "s"
            inputs = self.molmo_processor.process(
                images=[image],
                text="point to the " + object_prompt,
            )
            with torch.no_grad():
                inputs = {
                    k: v.to(self.molmo_model.device).unsqueeze(0)
                    for k, v in inputs.items()
                }
                output = self.molmo_model.generate_from_batch(
                    inputs,
                    GenerationConfig(max_new_tokens=200, stop_strings="<|endoftext|>"),
                    tokenizer=self.molmo_processor.tokenizer,
                )
                generated_tokens = output[0, inputs["input_ids"].size(1) :]
                generated_text = self.molmo_processor.tokenizer.decode(
                    generated_tokens, skip_special_tokens=True
                )
                pts = self._extract_points(generated_text, image.size[0], image.size[1])

        if len(pts) == 0:
            self.write_trace(f"<p> No points found<p>")
            return []

        # trace
        if self.oracle:
            self.write_trace(f"<p>Locate [Oracle]: {original_object_prompt}<p>")
        else:
            self.write_trace(f"<p>Locate: {original_object_prompt}<p>")
        dotted_im = dotted_image(image, pts)
        dotted_html = html_embed_image(dotted_im)
        self.write_trace(dotted_html)
        if len(pts) > 1 and original_object_prompt[-1] != 's':
            original_object_prompt += 's'
        self.write_trace(f"<p>{len(pts)} {original_object_prompt} found<p>")
        self.write_trace(f"<p>Points: {pts}<p>")
        return pts

    def execute_bboxs(self, image, object_prompt):
        original_object_prompt = object_prompt
        width, height = image.size
        prompt = f"{object_prompt.replace(' ', '-')} ."
        _, img_gd = self.transform_image(image)

        with torch.autocast(device_type="cuda", enabled=True, dtype=torch.float16):
            boxes, logits, phrases = predict(
                model=self.grounding_dino,
                image=img_gd,
                caption=prompt,
                box_threshold=self.BOX_THRESHOLD,
                text_threshold=self.TEXT_TRESHOLD,
                device="cuda:0",
            )
        bboxes = self._parse_bounding_boxes(boxes, width, height)

        if len(bboxes) == 0:
            self.write_trace(f"<p> No objects found<p>")
            return []

        # trace
        if self.oracle:
            self.write_trace(f"<p>Locate [Oracle]: {original_object_prompt}<p>")
        else:
            self.write_trace(f"<p>Locate: {original_object_prompt}<p>")
        boxed_image = box_image(image, bboxes)
        boxed_html = html_embed_image(boxed_image)
        self.write_trace(boxed_html)
        if len(bboxes) > 1 and original_object_prompt[-1] != 's':
            original_object_prompt += 's'
        self.write_trace(f"<p>{len(bboxes)} {original_object_prompt} found<p>")
        self.write_trace(f"<p>Boxes: {bboxes}<p>")

        return bboxes


class VQAModule(OracleModule):
    def __init__(
        self,
        dataset="omni3d",
        sam2_predictor=None,
        device=None,
        trace_path=None,
        api_key_path="./api.key",
    ):
        super().__init__("vqa", trace_path)
        self.generator = Generator("gpt-4o", api_key_path=api_key_path)
        self.dataset = dataset

        if self.dataset in ["clevr", "gqa"]:
            self.sam2_predictor = sam2_predictor
            self.device = device

    def _get_prompt(self, question, holistic=False):
        if self.dataset == "clevr":
            return VQA_PROMPT_CLEVR.format(question=question)
        elif self.dataset == "gqa":
            if holistic:
                print("using gqa vqa prompt holistic")
                return VQA_PROMPT_GQA_HOLISTIC.format(question=question)
            else:
                return VQA_PROMPT_GQA.format(question=question)
        else:
            return VQA_PROMPT.format(question=question)

    def _get_bbox(self, mask, margin=20):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Add margin
        rmin = max(0, rmin - margin)
        cmin = max(0, cmin - margin)
        rmax = min(mask.shape[0] - 1, rmax + margin)
        cmax = min(mask.shape[1] - 1, cmax + margin)

        return [cmin, rmin, cmax, rmax]

    def execute_pts(self, image, question, x, y):
        if self.oracle:
            answer = self.oracle.answer_question(x, y, question, self.scene_json)
        else:
            if int(x) == 0 and int(y) == 0:
                answer = self.predict(image, question, holistic=True)
                boxed_image = image
            else:
                with torch.no_grad():
                    sam_inpt_pts = np.array([[int(x), int(y)]])
                    sam_inpt_label = np.array([1])  # foreground label
                    self.sam2_predictor.set_image(np.array(image))

                    masks, scores, logits = self.sam2_predictor.predict(
                        point_coords=sam_inpt_pts,
                        point_labels=sam_inpt_label,
                        multimask_output=True,
                    )

                sorted_ind = np.argsort(scores)[::-1]
                masks = masks[sorted_ind]
                scores = scores[sorted_ind]
                if scores[1] > 0.3:
                    box1 = self._get_bbox(masks[0])
                    box2 = self._get_bbox(masks[1])
                    box = [
                        min(box1[0], box2[0]),
                        min(box1[1], box2[1]),
                        max(box1[2], box2[2]),
                        max(box1[3], box2[3]),
                    ]
                else:
                    box = self._get_bbox(masks[0])
                boxed_image = box_image(image, [box])
                answer = self.predict(boxed_image, question)

        # trace
        im_html = html_embed_image(image, 300)
        if self.oracle:
            self.write_trace(f"<p>Question [Oracle]: {question}</p>")
        else:
            self.write_trace(f"<p>Question: {question}</p>")
        if self.oracle:
            dotted_im = dotted_image(image, [[x, y]])
            dotted_im_html = html_embed_image(dotted_im, 300)
            self.write_trace(dotted_im_html)
        else:
            dotted_im = dotted_image(image, [[x, y]])
            dotted_im_html = html_embed_image(dotted_im, 300)
            self.write_trace(dotted_im_html)
        self.write_trace(f"<p>Answer: {answer}<p>")

        return answer.lower()

    def execute_bboxs(self, image, question, bbox):
        if bbox is None:
            answer = self.predict(image, question, holistic=True)
            boxed_image = image
        else:
            boxed_image = box_image(image, [bbox])
            answer = self.predict(boxed_image, question)

        im_html = html_embed_image(image, 300)
        self.write_trace(im_html)
        boxed_im_html = html_embed_image(boxed_image, 300)
        self.write_trace(boxed_im_html)
        self.write_trace(f"<p>{answer}<p>")
        return answer.lower()

    def remove_substring(self, output, substring):
        if substring in output:
            return output.replace(substring, "")
        else:
            return output

    def predict(self, img, question, holistic=False):
        prompt = self._get_prompt(question, holistic)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    },
                ],
            }
        ]
        output, _ = self.generator.generate("", messages)
        output = self.remove_substring(output, "```python")
        output = self.remove_substring(output, "```")
        answer = re.findall(r"<answer>(.*?)</answer>", output, re.DOTALL)[0].lower()
        return answer


class DepthModule(OracleModule):
    def __init__(
        self,
        unidepth_model,
        device,
        trace_path=None,
    ):
        super().__init__("depth", trace_path)
        self.unidepth_model = unidepth_model
        self.device = device

    def execute_pts(self, image, x, y):
        if self.oracle:
            depth = self.oracle.depth(x, y, self.scene_json)
        else:
            with torch.no_grad():
                rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(self.device)
                preds = self.unidepth_model.infer(rgb)["depth"].squeeze().cpu().numpy()
                depth = preds[int(y), int(x)]
        if self.oracle:
            self.write_trace(f"<p>Get Depth [Oracle]: ({x}, {y})<p>")
        else:
            self.write_trace(f"<p>Get Depth: ({x}, {y})<p>")
        dotted_im = dotted_image(image, [[x, y]])
        dotted_html = html_embed_image(dotted_im)
        self.write_trace(dotted_html)
        dotted_im = dotted_image(preds, [[x, y]])
        dotted_html = html_embed_image(dotted_im)
        self.write_trace(dotted_html)
        self.write_trace(f"<p>Depth: {depth}<p>")
        return depth

    def execute_bboxs(self, image, bbox):
        x_mid = (bbox[0] + bbox[2]) / 2
        y_mid = (bbox[1] + bbox[3]) / 2
        with torch.no_grad():
            rgb = torch.from_numpy(np.array(image)).permute(2, 0, 1).to(self.device)
            preds = self.unidepth_model.infer(rgb)["depth"].squeeze().cpu().numpy()
            depth = preds[int(y_mid), int(x_mid)]
        if self.oracle:
            self.write_trace(f"<p>Depth [Oracle]: ({x_mid}, {y_mid})<p>")
        else:
            self.write_trace(f"<p>Depth: ({x_mid}, {y_mid})<p>")
        dotted_im = dotted_image(image, [[x_mid, y_mid]])
        dotted_html = html_embed_image(dotted_im)
        self.write_trace(dotted_html)
        dotted_im = dotted_image(preds, [[x_mid, y_mid]])
        dotted_html = html_embed_image(dotted_im)
        self.write_trace(dotted_html)
        self.write_trace(f"<p>Depth: {depth}<p>")
        return depth


class SameObjectModule(OracleModule):
    def __init__(
        self, dataset="omni3d", sam2_predictor=None, device=None, trace_path=None
    ):
        super().__init__("same_object", trace_path)
        self.dataset = dataset

        if self.dataset in ["clevr", "gqa"]:
            self.sam2_predictor = sam2_predictor
            self.device = device

    def _get_bbox(self, mask, margin=20):
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        # Add margin
        rmin = max(0, rmin - margin)
        cmin = max(0, cmin - margin)
        rmax = min(mask.shape[0] - 1, rmax + margin)
        cmax = min(mask.shape[1] - 1, cmax + margin)

        return [cmin, rmin, cmax, rmax]

    def get_iou(self, box1, box2):
        # Coordinates of the intersection rectangle
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])

        # Width and height of the intersection rectangle
        width_inter = max(0, x2_inter - x1_inter)
        height_inter = max(0, y2_inter - y1_inter)

        # Area of the intersection
        area_inter = width_inter * height_inter

        # Area of both bounding boxes
        area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

        # Union of both bounding boxes
        area_union = area_box1 + area_box2 - area_inter

        # IoU calculation
        iou = area_inter / area_union if area_union != 0 else 0

        return iou

    def _get_mask(self, sam_inpt_pts, sam_inpt_label):
        with torch.no_grad():
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=sam_inpt_pts,
                point_labels=sam_inpt_label,
                multimask_output=True,
            )
            sorted_ind = np.argsort(scores)[::-1]
            masks = masks[sorted_ind]

        return masks[0]

    def execute_bboxs(self, image, bbox1, bbox2):
        answer = self.get_iou(bbox1, bbox2) > 0.92
        boxed_image = box_image(image, [bbox1, bbox2])
        im_html = html_embed_image(boxed_image, 300)
        self.write_trace(im_html)
        self.write_trace(f"<p>{answer}<p>")

        return answer

    def execute_pts(self, image, x_1, y_1, x_2, y_2):
        if self.oracle:
            answer = self.oracle.same_object(x_1, y_1, x_2, y_2, self.scene_json)
        else:
            sam_inpt_label = np.array([1])  # foreground label
            obj_1_sam_inpt_pts = np.array([[int(x_1), int(y_1)]])
            obj_2_sam_inpt_pts = np.array([[int(x_2), int(y_2)]])
            self.sam2_predictor.set_image(np.array(image))

            obj_1_mask = self._get_mask(obj_1_sam_inpt_pts, sam_inpt_label)
            obj_2_mask = self._get_mask(obj_2_sam_inpt_pts, sam_inpt_label)

            obj_1_bbox = self._get_bbox(obj_1_mask)
            obj_2_bbox = self._get_bbox(obj_2_mask)

            answer = self.get_iou(obj_1_bbox, obj_2_bbox) > 0.92

        if self.oracle:
            self.write_trace(f"<p>Same Object [Oracle]: ({x_1}, {y_1}) and ({x_2}, {y_2})<p>")
        else:
            self.write_trace(f"<p>Same Object: ({x_1}, {y_1}) and ({x_2}, {y_2})<p>")
        boxed_image = box_image(image, [obj_1_bbox, obj_2_bbox])
        im_html = html_embed_image(boxed_image, 300)
        self.write_trace(im_html)
        self.write_trace(f"<p>Answer: {answer}<p>")

        return answer


class Get2DObjectSize(PredefinedModule):
    def __init__(
        self, dataset="omni3d", sam2_predictor=None, device=None, trace_path=None
    ):
        super().__init__("get_2D_object_size", trace_path)
        self.dataset = dataset

        if self.dataset in ["clevr", "gqa"]:
            self.sam2_predictor = sam2_predictor
            self.device = device

    def execute_bboxs(self, image, bbox):
        width = abs(bbox[0] - bbox[2])
        height = abs(bbox[1] - bbox[3])

        # trace
        boxed_image = box_image(image, [bbox])
        boxed_im_html = html_embed_image(boxed_image, 300)
        self.write_trace(boxed_im_html)
        self.write_trace(f"<p>Width: {width}, Height: {height}<p>")

        return width, height

    def execute_pts(self, image, x, y):
        with torch.no_grad():
            sam_inpt_pts = np.array([[int(x), int(y)]])
            sam_inpt_label = np.array([1])  # foreground label
            self.sam2_predictor.set_image(np.array(image))

            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=sam_inpt_pts,
                point_labels=sam_inpt_label,
                multimask_output=True,
            )
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]
        if scores[1] > 0.3:
            box1 = self._get_bbox(masks[0])
            box2 = self._get_bbox(masks[1])
            box = [
                min(box1[0], box2[0]),
                min(box1[1], box2[1]),
                max(box1[2], box2[2]),
                max(box1[3], box2[3]),
            ]
        elif scores[2] > 0.2:
            box1 = self._get_bbox(masks[0])
            box2 = self._get_bbox(masks[1])
            box3 = self._get_bbox(masks[2])
            box = [
                min(box1[0], box2[0], box3[0]),
                min(box1[1], box2[1], box3[1]),
                max(box1[2], box2[2], box3[2]),
                max(box1[3], box2[3], box3[3]),
            ]
        else:
            box = self._get_bbox(masks[0])

        width = abs(box[0] - box[2])
        height = abs(box[1] - box[3])

        # trace
        if self.oracle:
            self.write_trace(f"<p>Get 2D Object Size [Oracle]: ({x}, {y})<p>")
        else:
            self.write_trace(f"<p>Get 2D Object Size: ({x}, {y})<p>")
        boxed_image = box_image(image, [box])
        boxed_im_html = html_embed_image(boxed_image, 300)
        self.write_trace(boxed_im_html)
        self.write_trace(f"<p>Width: {width}, Height: {height}<p>")

        return width, height


class ResultModule(PredefinedModule):
    def __init__(self, trace_path=None):
        super().__init__("result", trace_path)

    def execute_pts(self, var):
        self.write_trace(f"<p>Result: {var}<p>")
        return str(var)

    def execute_bboxs(self, var):
        self.write_trace(f"<p>Result: {var}<p>")
        return str(var)


class ModulesList:
    def __init__(self, models_path=None, trace_path=None, dataset="omni3d", api_key_path="./api.key"):
        set_devices()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dataset = dataset

        if dataset in ["clevr", "gqa"]:
            self.sam2_checkpoint = f"{models_path}/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
            self.sam2_model_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
            self.sam2_predictor = SAM2ImagePredictor(
                build_sam2(
                    self.sam2_model_cfg, self.sam2_checkpoint, device=self.device
                )
            )
            print("SAM2 Initialized")
            self.molmo_processor = AutoProcessor.from_pretrained(
                "allenai/Molmo-7B-D-0924",
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
            )
            self.molmo_model = AutoModelForCausalLM.from_pretrained(
                "allenai/Molmo-7B-D-0924",
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto",
            )
            print("Molmo Initialized")
        else:
            self.grounding_dino = load_model(
                f"{models_path}/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
                f"{models_path}/GroundingDINO/weights/groundingdino_swint_ogc.pth"
            )
            print("GroundingDINO Initialized")

        self.unidepth_model = UniDepthV2.from_pretrained(
            "lpiccinelli/unidepth-v2-vits14"
        ).to(self.device)

        self.modules = self.get_module_list(self.dataset, trace_path, api_key_path)
        self.module_names = [module.name for module in self.modules]
        self.module_executes = self.get_module_executes(self.dataset)

    def get_module_executes(self, dataset):
        if dataset in ["clevr", "gqa"]:
            return {
                self.module_names[i]: self.modules[i].execute_pts
                for i in range(len(self.modules))
            }
        else:
            return {
                self.module_names[i]: self.modules[i].execute_bboxs
                for i in range(len(self.modules))
            }

    def get_module_list(self, dataset, trace_path, api_key_path):
        if dataset in ["clevr", "gqa"]:
            return [
                LocateModule(
                    dataset=dataset,
                    molmo_processor=self.molmo_processor,
                    molmo_model=self.molmo_model,
                    trace_path=trace_path,
                ),
                VQAModule(
                    dataset=dataset,
                    sam2_predictor=self.sam2_predictor,
                    device=self.device,
                    trace_path=trace_path,
                    api_key_path=api_key_path,
                ),
                DepthModule(self.unidepth_model, self.device, trace_path),
                SameObjectModule(
                    dataset=dataset,
                    sam2_predictor=self.sam2_predictor,
                    device=self.device,
                    trace_path=trace_path,
                ),
                Get2DObjectSize(
                    dataset=dataset,
                    sam2_predictor=self.sam2_predictor,
                    device=self.device,
                    trace_path=trace_path,
                ),
                ResultModule(trace_path),
            ]
        else:
            return [
                LocateModule(
                    dataset=dataset,
                    grounding_dino=self.grounding_dino,
                    trace_path=trace_path,
                ),
                VQAModule(
                    dataset=dataset, trace_path=trace_path, api_key_path=api_key_path
                ),
                DepthModule(self.unidepth_model, self.device, trace_path),
                SameObjectModule(dataset=dataset, trace_path=trace_path),
                Get2DObjectSize(dataset=dataset, trace_path=trace_path),
                ResultModule(trace_path),
            ]

    def set_trace_path(self, trace_path):
        for module in self.modules:
            module.trace_path = trace_path

    def set_oracle(self, oracle, reference_image, scene_json):
        for module in self.modules:
            if hasattr(module, "set_oracle"):
                module.set_oracle(oracle, reference_image, scene_json)

    def clear_oracle(self):
        for module in self.modules:
            if hasattr(module, "set_oracle"):
                module.clear_oracle()
