# https://www.geeksforgeeks.org/python-convert-image-to-pdf-using-img2pdf-module/
# https://stackoverflow.com/questions/35859140/remove-transparency-alpha-from-any-image-using-pil
import img2pdf
from PIL import Image 
import os
import glob

name = "simple" # MODIFY NAME

script_dir = os.path.dirname(os.path.abspath(__file__))
file_matches = glob.glob(os.path.join(script_dir, f"{name}.*"))
img_path = None
for file in file_matches:
    if file.lower().endswith(("jpg", "png")):
        img_path = file
        break
pdf_path = os.path.join(script_dir, f"{name}.pdf")

try:
    image = Image.open(img_path)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[-1])
        image.close()
        temp_img_path = os.path.join(script_dir, "temp_image.png")
        bg.save(temp_img_path, format="PNG")
        bg.close()
        image = Image.open(temp_img_path)
        pdf_bytes = img2pdf.convert(image.filename)
        image.close()
        os.remove(temp_img_path)

    else:
        pdf_bytes = img2pdf.convert(image.filename)
        image.close()

    file = open(pdf_path, "wb")
    file.write(pdf_bytes)
    file.close()

except FileNotFoundError:
        print(f"Error: The file '{img_path}' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
