from models import SimpleModel
from model_trainer import train
from dataset import HeartDiseaseDataset

dataset = HeartDiseaseDataset("heart.csv")
train_loader, test_loader = dataset.get_data_loaders()

model = SimpleModel()

train(network=model, test_loader=test_loader, train_loader=train_loader, num_epochs=100, plot_loss=True, save_model=True)