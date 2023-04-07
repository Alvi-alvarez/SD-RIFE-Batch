import gradio as gr
from modules import scripts, shared, deepbooru, script_callbacks
from modules.processing import process_images, StableDiffusionProcessingImg2Img, Processed
import sys
import torch
import os
from torch.nn import functional as F
import numpy as np
import moviepy.video.io.ImageSequenceClip as ImageSequenceClip
from PIL import Image
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from rife.RIFE_HDv3 import Model


def rife_tab():
    with gr.Column():
        with gr.Row():
            with gr.Column():
                path = gr.Textbox(label="Images directory",
                                  placeholder=r"C:\Users\dude\Desktop\images")
                rife_passes = gr.Number(label='RIFE passes', value=1)
                fps = gr.Number(label='FPS', value=30)
                width = gr.Number(label='width', value=512)
                height = gr.Number(label='height', value=512)
                fps = gr.Number(label='FPS', value=30)
                loop = gr.Checkbox(label="loop")
                rife_drop = gr.Checkbox(label='Drop original frames',
                                        value=False)
                button = gr.Button("Generate", variant='primary')
            video = gr.Video()
    button.click(
        rife,
        inputs=[rife_passes, path, loop, fps, rife_drop, width, height],
        outputs=video)


def rife(rife_passes, path, loop, fps, rife_drop, width, height):
    dir = Path(path)
    images = Script.listFiles(dir)
    step_images = []

    for image_path in images:
        image = Image.open(image_path)
        step_images += [image]

    # RIFE (from https://github.com/vladmandic/rife)
    lead_inout = int(0)
    tgt_w, tgt_h = round(width * 1), round(height * 1)
    rifemodel = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def rifeload(model_path: str = os.path.dirname(os.path.abspath(__file__)) +
                 '/rife/flownet-v46.pkl',
                 fp16: bool = False):
        global rifemodel  # pylint: disable=global-statement
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            if fp16:
                torch.set_default_tensor_type(torch.cuda.HalfTensor)
        rifemodel = Model()
        rifemodel.load_model(model_path, -1)
        rifemodel.eval()
        rifemodel.device()

    def execute(I0, I1, n):
        global rifemodel  # pylint: disable=global-statement
        if rifemodel.version >= 3.9:
            res = []
            for i in range(n):
                res.append(
                    rifemodel.inference(I0, I1, (i + 1) * 1. / (n + 1), scale))
            return res
        else:
            middle = rifemodel.inference(I0, I1, scale)
            if n == 1:
                return [middle]
            first_half = execute(I0, middle, n=n // 2)
            second_half = execute(middle, I1, n=n // 2)
            if n % 2:
                return [*first_half, middle, *second_half]
            else:
                return [*first_half, *second_half]

    def pad(img):
        return F.pad(img, padding).half() if fp16 else F.pad(img, padding)

    rife_images = []
    if loop:
        rife_images = step_images
        back = rife_images[:-1]
        back = back[::-1]
        rife_images.extend(back)
    else:
        rife_images = step_images

    for i in range(int(rife_passes)):
        print(f"RIFE pass {i + 1}")
        if rifemodel is None:
            rifeload()
        print('Interpolating', len(rife_images), 'images')
        frame = rife_images[0]
        w, h = tgt_w, tgt_h
        scale = 1.0
        fp16 = False

        tmp = max(128, int(128 / scale))
        ph = ((h - 1) // tmp + 1) * tmp
        pw = ((w - 1) // tmp + 1) * tmp
        padding = (0, pw - w, 0, ph - h)

        buffer = []

        I1 = pad(
            torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(
                device, non_blocking=True).unsqueeze(0).float() / 255.)
        for frame in rife_images:
            I0 = I1
            I1 = pad(
                torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(
                    device, non_blocking=True).unsqueeze(0).float() / 255.)
            output = execute(I0, I1, 1)
            for mid in output:
                mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(
                    1, 2, 0)))
                buffer.append(np.asarray(mid[:h, :w]))
            if not rife_drop:
                buffer.append(np.asarray(frame))
        # for _i in range(buffer_frames): # fill ending frames
        #    buffer.put(frame)
        rife_images = buffer

    frames = [np.asarray(rife_images[0])] * lead_inout + [
        np.asarray(t) for t in rife_images
    ] + [np.asarray(rife_images[-1])] * lead_inout
    clip = ImageSequenceClip.ImageSequenceClip(frames, fps=fps)
    filename = "rife.mp4"
    clip.write_videofile(os.path.join(dir, filename),
                         verbose=False,
                         logger=None)
    return os.path.join(dir, filename)


def add_tab():
    with gr.Blocks(analytics_enabled=False) as ui:
        rife_tab()
    return [(ui, "RIFE", "RIFE")]


script_callbacks.on_ui_tabs(add_tab)


class Script(scripts.Script):

    def title(self):
        return "RIFE Batch"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        rife_passes = gr.Number(label='RIFE passes', value=1)
        fps = gr.Number(label='FPS', value=30)
        loop = gr.Checkbox(label="loop")
        rife_drop = gr.Checkbox(label='Drop original frames', value=False)
        path = gr.Textbox(label="Images directory",
                          placeholder=r"C:\Users\dude\Desktop\images")
        return [rife_passes, path, loop, fps, rife_drop]

    #(from https://github.com/yownas/seed_travel)
    def get_next_sequence_number(path):
        result = -1
        dir = Path(path)
        for file in dir.iterdir():
            if not file.is_dir(): continue
            try:
                num = int(file.name)
                if num > result: result = num
            except ValueError:
                pass
        return result + 1

    def listFiles(path):
        allFiles = [
            os.path.join(path, f) for f in os.listdir(path)
            if os.path.isfile(os.path.join(path, f)) and f.endswith((
                '.png', '.jpg', '.jpeg', '.bmp', '.PNG', '.JPG', '.JPEG',
                '.BMP'))
        ]
        return allFiles

    def run(self, p, rife_passes, path, loop, fps, rife_drop):
        all_images = []
        step_images = []
        images = Script.listFiles(path)
        p.outpath_samples = "outputs/RIFE"
        travel_path = os.path.join(p.outpath_samples, "")
        os.makedirs(travel_path, exist_ok=True)
        travel_number = Script.get_next_sequence_number(travel_path)
        travel_path = os.path.join(travel_path, f"{travel_number:05}")
        p.outpath_samples = travel_path

        for image_path in images:
            image = Image.open(image_path)
            img2img = StableDiffusionProcessingImg2Img(
                sd_model=shared.sd_model,
                prompt=p.prompt,
                outpath_samples=p.outpath_samples,
                negative_prompt=p.negative_prompt,
                seed=p.seed,
                sampler_name=p.sampler_name,
                steps=p.steps,
                cfg_scale=p.cfg_scale,
                width=p.width,
                height=p.height,
                restore_faces=p.restore_faces,
                tiling=False,
                init_images=[image],
                denoising_strength=p.denoising_strength,
                image_cfg_scale=p.image_cfg_scale,
            )
            img = process_images(img2img)
            all_images.append(img.images[0])
            step_images += [img.images[0]]

        # RIFE (from https://github.com/vladmandic/rife)
        lead_inout = int(0)
        tgt_w, tgt_h = round(p.width * 1), round(p.height * 1)  #fix
        rifemodel = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        def rifeload(
                model_path: str = os.path.dirname(os.path.abspath(__file__)) +
            '/rife/flownet-v46.pkl',
                fp16: bool = False):
            global rifemodel  # pylint: disable=global-statement
            torch.set_grad_enabled(False)
            if torch.cuda.is_available():
                torch.backends.cudnn.enabled = True
                torch.backends.cudnn.benchmark = True
                if fp16:
                    torch.set_default_tensor_type(torch.cuda.HalfTensor)
            rifemodel = Model()
            rifemodel.load_model(model_path, -1)
            rifemodel.eval()
            rifemodel.device()

        def execute(I0, I1, n):
            global rifemodel  # pylint: disable=global-statement
            if rifemodel.version >= 3.9:
                res = []
                for i in range(n):
                    res.append(
                        rifemodel.inference(I0, I1, (i + 1) * 1. / (n + 1),
                                            scale))
                return res
            else:
                middle = rifemodel.inference(I0, I1, scale)
                if n == 1:
                    return [middle]
                first_half = execute(I0, middle, n=n // 2)
                second_half = execute(middle, I1, n=n // 2)
                if n % 2:
                    return [*first_half, middle, *second_half]
                else:
                    return [*first_half, *second_half]

        def pad(img):
            return F.pad(img, padding).half() if fp16 else F.pad(img, padding)

        rife_images = []
        if loop:
            rife_images = step_images
            back = rife_images[:-1]
            back = back[::-1]
            rife_images.extend(back)
        else:
            rife_images = step_images

        for i in range(int(rife_passes)):
            print(f"RIFE pass {i + 1}")
            if rifemodel is None:
                rifeload()
            print('Interpolating', len(rife_images), 'images')
            frame = rife_images[0]
            w, h = tgt_w, tgt_h
            scale = 1.0
            fp16 = False

            tmp = max(128, int(128 / scale))
            ph = ((h - 1) // tmp + 1) * tmp
            pw = ((w - 1) // tmp + 1) * tmp
            padding = (0, pw - w, 0, ph - h)

            buffer = []

            I1 = pad(
                torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(
                    device, non_blocking=True).unsqueeze(0).float() / 255.)
            for frame in rife_images:
                I0 = I1
                I1 = pad(
                    torch.from_numpy(np.transpose(frame, (2, 0, 1))).to(
                        device, non_blocking=True).unsqueeze(0).float() / 255.)
                output = execute(I0, I1, 1)
                for mid in output:
                    mid = (((mid[0] * 255.).byte().cpu().numpy().transpose(
                        1, 2, 0)))
                    buffer.append(np.asarray(mid[:h, :w]))
                if not rife_drop:
                    buffer.append(np.asarray(frame))
            # for _i in range(buffer_frames): # fill ending frames
            #    buffer.put(frame)
            rife_images = buffer

        frames = [np.asarray(rife_images[0])] * lead_inout + [
            np.asarray(t) for t in rife_images
        ] + [np.asarray(rife_images[-1])] * lead_inout
        clip = ImageSequenceClip.ImageSequenceClip(frames, fps=fps)
        filename = f"rife-{travel_number:05}.mp4"
        clip.write_videofile(os.path.join(travel_path, filename),
                             verbose=False,
                             logger=None)
        #fix---------------------------
        res = Processed(p, all_images)
        print("Finished")
        shared.state.interrupt()
        #------------------------
        return res
