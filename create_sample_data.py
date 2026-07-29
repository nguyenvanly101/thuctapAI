import os
from PIL import Image, ImageDraw

# Create dataset subdirectories
base_dir = os.path.dirname(__file__)
dataset_dir = os.path.join(base_dir, 'dataset')
classes = ['hong_trung_quoc', 'hong_lang_son', 'hong_da_lat']

for cls in classes:
    os.makedirs(os.path.join(dataset_dir, cls), exist_ok=True)

def generate_persimmon_image(filename, color, shape_type="round", text=""):
    """Generates synthetic persimmon image for testing"""
    img = Image.new('RGB', (400, 400), color=(240, 243, 248))
    draw = ImageDraw.Draw(img)

    # Draw persimmon fruit body
    if shape_type == "square_bapped":
        # Chinese persimmon: Square/bapped
        draw.rounded_rectangle([80, 100, 320, 320], radius=50, fill=color, outline=(200, 60, 0), width=4)
        # Calyx / stem
        draw.polygon([(180, 90), (220, 90), (230, 110), (170, 110)], fill=(60, 120, 40))
    elif shape_type == "acorn_elongated":
        # Lang Son persimmon: Acorn / elongated
        draw.ellipse([110, 80, 290, 340], fill=color, outline=(180, 100, 0), width=4)
        # Calyx hugging the body
        draw.polygon([(170, 75), (230, 75), (250, 120), (150, 120)], fill=(40, 100, 30))
    else:
        # Da Lat persimmon: Oval / Egg shape
        draw.ellipse([90, 90, 310, 330], fill=color, outline=(190, 80, 0), width=4)
        # Natural calyx
        draw.polygon([(160, 80), (240, 80), (220, 110), (180, 110)], fill=(50, 110, 35))

    # Add shadow
    draw.ellipse([100, 335, 300, 360], fill=(200, 205, 215))
    
    img.save(filename)
    print(f"Generated sample: {filename}")

# Generate samples for Chinese Persimmon (Red-orange, square)
generate_persimmon_image(
    os.path.join(dataset_dir, 'hong_trung_quoc', 'sample_tq_01.jpg'),
    color=(255, 70, 20),
    shape_type="square_bapped"
)

# Generate samples for Lang Son Persimmon (Yellow-orange, acorn)
generate_persimmon_image(
    os.path.join(dataset_dir, 'hong_lang_son', 'sample_ls_01.jpg'),
    color=(245, 160, 20),
    shape_type="acorn_elongated"
)

# Generate samples for Da Lat Persimmon (Natural orange, oval)
generate_persimmon_image(
    os.path.join(dataset_dir, 'hong_da_lat', 'sample_dl_01.jpg'),
    color=(255, 115, 25),
    shape_type="oval_egg"
)

print("Sample dataset initialized successfully!")
