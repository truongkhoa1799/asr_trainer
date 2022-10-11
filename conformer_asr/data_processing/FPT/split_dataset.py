import argparse
from sklearn.model_selection import train_test_split
from conformer_asr.utils import read_manifest, Logger, save_manifest

LOGGER = Logger(name="SPLIT_TRAINING_TESTING_MANIFESTS_FPT")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest_path', default='', help='name of evaluation dataset')
    parser.add_argument('--train_manifest_path', default='', help='name of evaluation dataset')
    parser.add_argument('--test_manifest_path', default='', help='name of evaluation dataset')
    args = parser.parse_args()
    try:
        manifest_datas, duration = read_manifest(args.manifest_path)
        LOGGER.log_info(f"Spliting {len(manifest_datas)} data with {round(duration // 3600, 2)} hours to training and testing data")
        train_manifest_datas, test_manifest_datas = train_test_split(manifest_datas, test_size=0.05, random_state=42)
        
        save_manifest(args.train_manifest_path, train_manifest_datas)
        save_manifest(args.test_manifest_path, test_manifest_datas)
        
        LOGGER.log_info(f"Number of training data: {len(train_manifest_datas)} saved as {args.train_manifest_path}")
        LOGGER.log_info(f"Number of training data: {len(test_manifest_datas)} saved as {args.test_manifest_path}")
    except Exception as e:
        print(e)
        exit(-1)
    