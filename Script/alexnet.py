import os
import requests
from io import BytesIO
from tkinter import Tk, filedialog
from tqdm import tqdm
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import shutil
import time


# Define the data transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Prompt the user to select the data directory
Tk().withdraw()
data_dir = filedialog.askdirectory(title="Select the data directory")
model_path = os.path.join(data_dir, "alexnet_custom_model.pt")

def create_train_test_folders(root_folder, split_percentage):
    # Get a list of subdirectories in root_folder
    class_names = [name for name in os.listdir(root_folder) if not name.startswith('.') and os.path.isdir(os.path.join(root_folder, name))]

    # Make train and valid directories in root_folder
    train_dir = os.path.join(root_folder, 'train')
    os.makedirs(train_dir, exist_ok=True)
    valid_dir = os.path.join(root_folder, 'valid')
    os.makedirs(valid_dir, exist_ok=True)

    # Iterate over class_names and create train and valid directories for each
    for class_name in class_names:
        class_train_dir = os.path.join(train_dir, class_name)
        os.makedirs(class_train_dir, exist_ok=True)
        class_valid_dir = os.path.join(valid_dir, class_name)
        os.makedirs(class_valid_dir, exist_ok=True)

        # Get a list of all files in the class_name directory (excluding directories)
        class_dir = os.path.join(root_folder, class_name)
        all_files = [name for name in os.listdir(class_dir) if
                     not name.startswith('.') and os.path.isfile(os.path.join(class_dir, name))]

        # Shuffle the list of files and split them based on the split_percentage
        random.shuffle(all_files)
        split_index = int(len(all_files) * split_percentage)
        train_files = all_files[:split_index]
        valid_files = all_files[split_index:]

        # Move the files to their respective train and valid directories
        for train_file in train_files:
            src_file = os.path.join(class_dir, train_file)
            dst_file = os.path.join(class_train_dir, train_file)
            shutil.copy(src_file, dst_file)
        for valid_file in valid_files:
            src_file = os.path.join(class_dir, valid_file)
            dst_file = os.path.join(class_valid_dir, valid_file)
            shutil.copy(src_file, dst_file)

def print_model_stats(model_path, data_dir):
    # Check if the model file exists
    if not os.path.exists(model_path):
        print("No model exists. Try training a new model.")
        return

    # Load the model
    model = torch.load(model_path)
    model.to(device)

    # Load the train and validation datasets
    train_dir = os.path.join(data_dir, 'train')
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    valid_dir = os.path.join(data_dir, 'valid')
    valid_dataset = datasets.ImageFolder(valid_dir, transform=transform)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=False)

    # Evaluate the model on the train set
    model.eval()
    train_correct = 0
    train_total = 0
    with torch.no_grad(), tqdm(train_loader, desc='Evaluating on train set') as train_bar:
        for images, labels in train_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        train_acc = train_correct / train_total
        print('Training Accuracy: {:.2f}%'.format(train_acc * 100))

    # Evaluate the model on the validation set
    valid_correct = 0
    valid_total = 0
    with torch.no_grad(), tqdm(valid_loader, desc='Evaluating on validation set') as valid_bar:
        for images, labels in valid_bar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            valid_total += labels.size(0)
            valid_correct += (predicted == labels).sum().item()
        valid_acc = valid_correct / valid_total
        print('Validation Accuracy: {:.2f}%'.format(valid_acc * 100))

def classify():
    model = torch.load(model_path)
    model.to(device)
    train_dir = os.path.join(data_dir, 'train')
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)
    classify_image(model, train_dataset, device)


def classify_image(model, train_dataset, device):
    # Ask the user if they want to enter a URL or pick an image file from the computer
    print("How do you want to select the image to classify?")
    print("1. Enter URL")
    print("2. Pick from computer")
    choice = input("Enter your choice: ")

    # Load and preprocess the image
    if choice == "1":
        image_path = input("Enter the URL of the image: ")
        img_batch = load_image(image_path, url=True, transform=transform)
    elif choice == "2":
        Tk().withdraw()
        image_path = filedialog.askopenfilename(title="Select the image to classify", filetypes=(("JPEG files", "*.jpg"), ("PNG files", "*.png"), ("All files", "*.*")))
        if not image_path:
            print("No image selected.")
            return
        img_batch = load_image(image_path, transform=transform)
    else:
        print("Invalid choice.")
        return

    # Classify the image
    with torch.no_grad():
        outputs = model(img_batch)
        _, predicted = torch.max(outputs.data, 1)

    # Print the predicted class label
    # Print out the class names
    class_names = train_dataset.classes
    print("Classes found:", class_names)
    print(f"Predicted class: {class_names[predicted.item()]}")

def load_image(image_path, url=False, transform=None):
    if url:
        response = requests.get(image_path)
        img = Image.open(BytesIO(response.content)).convert('L')
    else:
        img = Image.open(image_path).convert('L')
    img = img.convert("RGB")  # Duplicate grayscale channel to create 3-channel image
    if transform:
        img_transformed = transform(img)
    else:
        img_transformed = transforms.ToTensor()(img)
    img_batch = img_transformed.unsqueeze(0)
    if device:
        img_batch = img_batch.to(device)
    return img_batch

def train_model(data_dir, device):
    # Load the pretrained AlexNet model
    model = models.alexnet(pretrained=True)

    # Load the custom dataset
    train_dir = os.path.join(data_dir, 'train')
    if not os.path.exists(train_dir):
        split = float(input("Enter the split percentage for the data (train/test): "))
        create_train_test_folders(data_dir, split / 100)
    train_dataset = datasets.ImageFolder(train_dir, transform=transform)

    # Update the last fully connected layer to match the number of classes in the custom dataset
    num_classes = len(train_dataset.classes)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)

    # Print out the class names
    class_names = train_dataset.classes
    print("Classes found:", class_names)

    # Set up the model, criterion, and optimizer
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Define the file path for the saved model
    model_path = os.path.join(data_dir, "alexnet_custom_model.pt")

    valid_dir = os.path.join(data_dir, 'valid')
    if not os.path.exists(valid_dir):
        create_train_test_folders(data_dir, 0.8)

    # Load the validation dataset
    valid_dataset = datasets.ImageFolder(valid_dir, transform=transform)

    # Create data loaders
    print("Creating data loaders...")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64, shuffle=False)
    print("Data loaders created successfully!")

    # Train the model
    num_epochs = int(input("Enter the number of epochs to train: "))
    print("Training model...")
    best_model = None
    best_valid_acc = 0.0
    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        epoch_start_time = time.time()
        for i, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss /= len(train_loader)
        accuracy = correct / total
        epoch_elapsed_time = time.time() - epoch_start_time
        print(
            'Epoch [{}/{}], Training Loss: {:.4f}, Training Accuracy: {:.2f}%, Elapsed Time: {:.2f}s'.format(epoch + 1,
                                                                                                             num_epochs,
                                                                                                             epoch_loss,
                                                                                                             accuracy * 100,
                                                                                                             epoch_elapsed_time))

        # Evaluate the model on the validation set
        model.eval()
        valid_correct = 0
        valid_total = 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                valid_total += labels.size(0)
                valid_correct += (predicted == labels).sum().item()
            valid_acc = valid_correct / valid_total
            print('Validation Accuracy: {:.2f}%'.format(valid_acc * 100))

            # Check if the current model has the best validation accuracy so far
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                best_model = model
                print('Best model updated. Validation Accuracy: {:.2f}%'.format(valid_acc * 100))
        # Save the best model to the file path
        if best_model is not None:
            torch.save(best_model, model_path)
        print('Best model saved to: {}'.format(model_path))

        epoch_elapsed_time = time.time() - epoch_start_time
        print(f"Epoch training time: {epoch_elapsed_time:.2f} seconds\n")

    elapsed_time = time.time() - start_time
    print(f"Training complete. Total training time: {elapsed_time:.2f} seconds")



action = int(input("Select an action:\n1. Train model\n2. Classify image\n3. Print model stats\nEnter action number: "))

if action == 1:
    # Train the model
    train_model(data_dir, device)
    shouldClassify = input("Do you also want to classify a new image? (y/n): ")
    if shouldClassify == "y" or shouldClassify == "Y":
        classify()
elif action == 2:
    # Classify an image
    # Load the model
    if not os.path.exists(model_path):
        print("Model not found. Please train a model first.")
        train_model(data_dir, device)
        classify()
    else:
        classify()
elif action == 3:
    # Print model stats
    model_path = os.path.join(data_dir, "alexnet_custom_model.pt")
    print_model_stats(model_path, data_dir)
else:
    print("Invalid action selected.")
