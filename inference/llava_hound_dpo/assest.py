from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

image_paths = ["test_images/52bffa01b64241c88c2ab4aad3c3fdb0.png", "test_images/aa71fb24609c4b86bd4a206cd314abef.png", "test_images/631293b2cc7f46188fd92ac9c7e6051b.png"]

original_image_np = [np.array(Image.open("/scratch/spp9399/MIA-DPO/gen_data/gen_instruction/" + p)) for p in image_paths]
min_height = min(img.shape[0] for img in original_image_np)
images_np_resized = [img[:min_height, :, :] for img in original_image_np]
merged_image_np = np.concatenate(images_np_resized, axis=1)

plt.figure(figsize=(10, 10))
plt.imshow(merged_image_np)
plt.title('Original Image')
plt.axis('off')
plt.savefig("/scratch/spp9399/original_image.png", dpi=900)
plt.close()

