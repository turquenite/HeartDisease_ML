from models import SimpleModel
from model_trainer import train
from dataset import HeartDiseaseDataset

dataset = HeartDiseaseDataset("heart.csv")
train_loader, test_loader = dataset.get_data_loaders()

model = SimpleModel()

train(network=model, eval_loader=test_loader, train_loader=train_loader, plot_loss=True)


