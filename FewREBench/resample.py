from torch.utils.data import DataLoader, Dataset
import json
from torchsampler import ImbalancedDatasetSampler
import numpy as np
import argparse
import os

class CustomDataset(Dataset):
    def __init__(self, datapath, rel_file):
        with open(datapath, 'r') as f:
            lines = f.readlines()
        self.data = []
        notstrdata = []
        self.x_data = []
        self.y_data = []
        for line in lines:
            self.data.append(line.strip())
            notstrdata.append(json.loads(line.strip()))
        with open(rel_file, 'r') as f:
            ids = json.loads(f.readlines()[0].strip())
        for example in notstrdata:
            self.y_data.append(ids[example.pop('relation')])
            self.x_data.append(str(example))

    def __getitem__(self, index):
        return self.data[index]
    
    def __len__(self):
        return len(self.data)
    
    def get_labels(self):
        return self.y_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", "-i", type=str, required=True,
                        help="The path of the training file.")
    parser.add_argument("--output_dir", "-o", type=str, required=True,
                        help="The directory of the sampled files.")
    parser.add_argument('--rel_file','-r',type=str, required=True,
                        help="the path of the relation file")

    args = parser.parse_args()
    os.makedirs(args.output_dir,exist_ok=True)
    dataset = CustomDataset(args.input_file,args.rel_file)
    for i in range(1,6):
        np.random.seed(i)
        dataLoader = DataLoader(dataset, batch_size=1,sampler=ImbalancedDatasetSampler(dataset))
        with open(os.path.join(args.output_dir, 'sa_'+str(i)+'.json'),'w') as f:
            for data in dataLoader:
                reldata = eval(data[0])
                f.writelines(json.dumps(reldata,ensure_ascii=False))
                f.write('\n')
