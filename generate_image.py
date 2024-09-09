from diffusers import StableDiffusionPipeline
import torch

# Enter the text prompt
prompt = input("Enter the text prompt for the image: ")
output_file = input("Enter the filename to save the image (e.g., 'image.png'): ")

print("Loading the Stable Diffusion model...")

# Load the model with CPU support and ensure it uses float32 (default for CPU)
pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", 
    torch_dtype=torch.float32  # Set to float32 for CPU
).to("cpu")  # Move the model to CPU

print("Generating image for prompt:", prompt)

# Generate the image
image = pipe(prompt).images[0]

# Save the image
image.save(output_file)
print(f"Image saved as {output_file}")
