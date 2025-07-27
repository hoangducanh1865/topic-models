from src.data.download_20ng import download_save
from src.preprocess.preprocess import Preprocess
from src.data.basic_dataset import BasicDataset
from src.model.basic.ProdLDA import ProdLDA
from src.trainer.basic.BasicTrainer import BasicTrainer
from src.config.config import DEVICE

def main():
    data_dir = "./data"
    download_save("./data")

    preprocess = Preprocess(vocab_size=2000, verbose=True)
    preprocess.preprocess_jsonlist(data_dir, label_name='group')

    dataset = BasicDataset(dataset_dir=data_dir, batch_size=128, device=DEVICE, read_labels=True)

    model = ProdLDA(vocab_size=dataset.vocab_size, num_topics=20, en_units=100, dropout=0.2).to(DEVICE)
    trainer = BasicTrainer(model=model, dataset=dataset, epochs=10, batch_size=128, num_top_words=10, verbose=True)
    top_words, train_theta = trainer.train()

    for i, words in enumerate(top_words):
        print(f"Topic {i}: {words}")

if __name__ == "__main__":
    main()