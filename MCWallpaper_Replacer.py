from PIL import Image, ImageChops
import cv2
import numpy as np
import torch
import importlib
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

void = (0, 0, 0, 255)

def generate_raw_texture(width, height, output_path='tex/random_texture.png'):
    if width > 255 or height > 255:
        raise ValueError('Image dimensions should be <=255')
    image = Image.new('RGB', (width, height))
    pixels = image.load()
    for y in range(height):
        for x in range(width):
            pixels[x, y] = (64, x * int(255/width), y * int(255/height))
    image.save(output_path)


def highlight_player(mat_color_path, layer_alpha_path, output_path):
    mat = Image.open(mat_color_path).convert('RGBA')
    alpha = Image.open(layer_alpha_path).convert('RGBA')
    if mat.size != alpha.size:
        raise ValueError(f'Mismatched sizes: {mat.size} vs {alpha.size}')
    pm, pa = mat.load(), alpha.load()
    w, h = mat.size
    for y in range(h):
        for x in range(w):
            if pa[x, y] != void:
                pa[x, y] = pm[x, y]
    alpha.save(output_path)


def replace_texture(highlighted_path, texture_a_path, texture_b_path, output_path):
    hp = Image.open(highlighted_path).convert('RGBA')
    ta = Image.open(texture_a_path).convert('RGBA')
    tb = Image.open(texture_b_path).convert('RGBA')
    if ta.size != tb.size:
        raise ValueError('Texture size mismatch')
    arr_hp = torch.from_numpy(np.array(hp)).to(device)
    arr_a = torch.from_numpy(np.array(ta)).to(device)
    arr_b = torch.from_numpy(np.array(tb)).to(device)
    uniq = torch.unique(arr_hp.view(-1, 4), dim=0)
    res = torch.zeros_like(arr_hp)
    for c in uniq:
        cnt = tuple(c.tolist())
        if cnt in (void, (0,0,0,0)):
            continue
        mask_a = (arr_a == c).all(-1)
        rep = arr_b[mask_a][0] if mask_a.any() else torch.tensor(c, device=device)
        mask_hp = (arr_hp == c).all(-1)
        res[mask_hp] = rep
    Image.fromarray(res.cpu().numpy().astype('uint8'), 'RGBA').save(output_path)


def replace_eyes(highlighted_path, default_colors, new_colors, output_path):
    src = Image.open(highlighted_path).convert('RGBA')
    w, h = src.size
    out_img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    pix = src.load()
    for y in range(h):
        for x in range(w):
            c = pix[x, y]
            if c in default_colors:
                out_img.putpixel((x, y), new_colors[default_colors.index(c)])
    out_img.save(output_path)


def composite_image(mat_path, light_path, output_path):
    m = Image.open(mat_path).convert('RGBA')
    l = Image.open(light_path).convert('RGBA')
    comp = ImageChops.multiply(m, l)
    comp.save(output_path)


def edge_detection(input_path, output_path, lower_threshold=50, upper_threshold=200,
                   edge_color=(255,255,255,255), dark_threshold=255, filling_area=0):
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    b,g,r,a = cv2.split(img)
    use_cuda = cv2.cuda.getCudaEnabledDeviceCount() > 0
    def canny(ch):
        if use_cuda:
            gm = cv2.cuda_GpuMat(); gm.upload(ch)
            det = cv2.cuda.createCannyEdgeDetector(lower_threshold, upper_threshold)
            return det.detect(gm).download()
        return cv2.Canny(ch, lower_threshold, upper_threshold)
    edges = [canny(ch) for ch in (b, g, r, a)]
    cs = b.astype('int16') + g.astype('int16') + r.astype('int16')
    mask = np.zeros_like(cs, bool)
    if filling_area == 1:
        mask = cs > dark_threshold
    elif filling_area == -1:
        mask = cs < dark_threshold
    mask &= (a != 0)
    out = np.zeros_like(img)
    for e in (*edges, mask):
        out[e != 0] = edge_color
    cv2.imwrite(output_path, out)


def MCWallpaper_Replace(wallpaper_folder, outputfolder, your_skin, have_eyes=1,
                        eye_color_file='eye_color_player', casu=0):
    os.makedirs(outputfolder, exist_ok=True)
    tex_dir = os.path.join(os.getcwd(), 'tex'); os.makedirs(tex_dir, exist_ok=True)
    try:
        w, h = Image.open(your_skin).size
    except:
        w, h = 64, 64
    generate_raw_texture(w, h, os.path.join(tex_dir, 'random_texture.png'))

    layers = {
        'body_color': 'layer_matcolor0000.png',
        'hat_color': 'layer_matcolor0001.png',
        'body_alpha': 'layer_object_1_0000.png',
        'hat_alpha': 'layer_object_1_0001.png',
        'eyes_alpha': 'layer_object_2_0000.png',
        'illum_body': 'layer_illum0000.png',
        'ao_body': 'layer_ao0000.png',
        'illum_hat': 'layer_illum0001.png',
        'ao_hat': 'layer_ao0001.png',
        'background': 'background0000.png'
    }
    out = lambda name: os.path.join(outputfolder, f"{name}.png")

    if have_eyes:
        eyes_mod = importlib.import_module(eye_color_file)
        def_mod = importlib.import_module('eye_colors')
        highlight_player(
            os.path.join(wallpaper_folder, layers['body_color']),
            os.path.join(wallpaper_folder, layers['eyes_alpha']),
            out('highlighted_eyes')
        )
        replace_eyes(
            out('highlighted_eyes'),
            def_mod.colors_default,
            eyes_mod.colors,
            out('changed_eyes')
        )
        composite_image(
            out('changed_eyes'),
            os.path.join(wallpaper_folder, layers['illum_body']),
            out('changed_eyes.illum')
        )
        composite_image(
            out('changed_eyes.illum'),
            os.path.join(wallpaper_folder, layers['ao_body']),
            out('changed_eyes.shadow')
        )

    for part, ck, ak, ik, ak2 in [
        ('body', 'body_color', 'body_alpha', 'illum_body', 'ao_body'),
        ('hat', 'hat_color', 'hat_alpha', 'illum_hat', 'ao_hat')
    ]:
        highlight_player(
            os.path.join(wallpaper_folder, layers[ck]),
            os.path.join(wallpaper_folder, layers[ak]),
            out(f'highlighted_{part}')
        )
        replace_texture(
            out(f'highlighted_{part}'),
            os.path.join(tex_dir, 'random_texture.png'),
            your_skin,
            out(f'changed_{part}')
        )
        composite_image(
            out(f'changed_{part}'),
            os.path.join(wallpaper_folder, layers[ik]),
            out(f'changed_{part}.illum')
        )
        composite_image(
            out(f'changed_{part}.illum'),
            os.path.join(wallpaper_folder, layers[ak2]),
            out(f'changed_{part}.shadow')
        )

    for phase in ['', '.illum', '.shadow']:
        canv = Image.new('RGBA', Image.open(out(f'changed_body{phase}')).size, (0,0,0,0))
        files = []
        if have_eyes:
            files.append(out(f'changed_eyes{phase}'))
        files.extend([out(f'changed_body{phase}'), out(f'changed_hat{phase}')])
        for fpath in files:
            part = Image.open(fpath).convert('RGBA')
            canv.paste(part, (0,0), part)
        canv.save(out(f'full{phase}'))
    base = Image.open(os.path.join(wallpaper_folder, layers['background'])).convert('RGBA')
    for layer in [out('changed_body.shadow'), out('changed_hat.shadow')] + ([out('changed_eyes.shadow')] if have_eyes else []):
        img_part = Image.open(layer).convert('RGBA')
        base.paste(img_part, (0,0), img_part)
    base.save(out('full_image'))
    for phase in ['', '.shadow']:
        edge_detection(
            out(f'full{phase}'),
            out(f'edge{phase}'),
            lower_threshold=50,
            upper_threshold=200,
            edge_color=(255,255,255,255),
            dark_threshold=int(255*0.3)
        )

    print('Done! Enjoy your wallpaper :D')
