from PIL import Image
import torch
import matplotlib.pyplot as plt
import os


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0) 


def predict_image(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        _, predicted_class = torch.max(output, 1)
    return predicted_class


def show_predicted_pokemon_image(pokemon_name, dataset_root):
    pokemon_image_path = os.path.join(dataset_root, pokemon_name)
    
    if os.path.exists(pokemon_image_path):
        image_files = [f for f in os.listdir(pokemon_image_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        if image_files:
            image_path = os.path.join(pokemon_image_path, image_files[0])
            image = Image.open(image_path)
            
            # Вывод изображения
            plt.imshow(image)
            plt.title(f"Predicted Pokemon: {pokemon_name}")
            plt.axis('off')
            plt.show()
        else:
            print(f"No images found for {pokemon_name}")
    else:
        print(f"Folder not found for {pokemon_name}")


def get_class_name(predicted_class, class_names):
    return class_names[predicted_class]
