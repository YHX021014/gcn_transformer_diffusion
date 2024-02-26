from dataset import *
import pickle

for desired_source in ['eth', 'hotel', 'univ', 'zara1', 'zara2']:
    for data_class in ['train', 'val', 'test']:
        obs_seq_len = 8
        pred_seq_len = 12
        gpu_num = 0
        data_set = os.path.join('./data/raw_data', desired_source)
        print(data_set)
        dataset = TrajectoryDataset(
            os.path.join(data_set , data_class),
            obs_len=obs_seq_len,
            pred_len=pred_seq_len,
            skip=1)

        data_dict_path = './data/processed/{}_{}.pkl'.format(desired_source, data_class)

        with open(data_dict_path, 'wb') as f:
            pickle.dump(dataset, f, protocol=4)

print('dataset completed! ')
