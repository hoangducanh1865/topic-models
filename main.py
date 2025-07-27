from src.data.download_20ng import download_save
from src.preprocess.preprocess import Preprocess
from src.data.basic_dataset import BasicDataset
from src.model.basic.ProdLDA import ProdLDA
from src.trainer.basic.BasicTrainer import BasicTrainer
from src.config.config import DEVICE
from src.eva.topic_coherence import _coherence
from src.eva.topic_diversity import _diversity
from src.eva.classification import _cls
from src.eva.clustering import _clustering

def main():
    data_dir = "./data"
    download_save("./data")

    preprocess = Preprocess(vocab_size=2000, verbose=True)
    rst = preprocess.preprocess_jsonlist(data_dir, label_name='group')
    preprocess.save(
        data_dir,
        vocab=rst['vocab'],
        train_texts=rst['train_texts'],
        train_bow=rst['train_bow'],
        train_labels=rst.get('train_labels'),
        test_texts=rst.get('test_texts'),
        test_bow=rst.get('test_bow'),
        test_labels=rst.get('test_labels')
    )

    dataset = BasicDataset(dataset_dir=data_dir, batch_size=128, device=DEVICE, read_labels=True)

    model = ProdLDA(vocab_size=dataset.vocab_size, num_topics=20, en_units=100, dropout=0.2).to(DEVICE)
    trainer = BasicTrainer(model=model, dataset=dataset, epochs=100, batch_size=128, num_top_words=10, verbose=True)
    top_words, train_theta = trainer.train()

    for i, words in enumerate(top_words):
        print(f"Topic {i}: {words}")
        
    TC = _coherence(dataset.train_texts, dataset.vocab, top_words)
    print(f"TC: {TC:.5f}")
    
    TD = _diversity(top_words)    
    print(f"TD: {TD:.5f}")

    train_theta, test_theta = trainer.export_theta()
    
    clt_results = _clustering(test_theta, dataset.test_labels)
    print(clt_results)

    cls_results = _cls(train_theta, test_theta, dataset.train_labels, dataset.test_labels)
    print(cls_results)


if __name__ == "__main__":
    main()